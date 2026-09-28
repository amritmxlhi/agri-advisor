"""
Module 4: Leaf Disease Detection — full training pipeline
-----------------------------------------------------------
Transfer learning (ImageNet-pretrained CNN) on a folder-per-class leaf
dataset such as PlantVillage.

What this script does:
  1. Stratified train/val/test split from one image folder
  2. Phase 1: train only the new classifier head (backbone frozen)
  3. Phase 2: unfreeze everything, fine-tune with a small LR + cosine decay
  4. Class-weighted, label-smoothed loss (PlantVillage is imbalanced)
  5. Early stopping / best checkpoint chosen on validation macro-F1
  6. Temperature scaling on the validation set so the reported confidence
     means something (raw CNN softmax scores are over-confident)
  7. Test report: accuracy, top-3, macro-F1, per-class report, confusion
     matrix, ECE before/after calibration
  8. Optional out-of-distribution check on real field photos (--field_dir)
  9. Export: checkpoint (.pt) + TorchScript (.ts.pt) + class_names.json

IMPORTANT caveat about PlantVillage: it contains many near-duplicate photos
of the same leaf, so a random split leaks and test accuracy is inflated
(commonly 98-99%+). The honest number is the field-image score; use
--field_dir with PlantDoc-style images to measure the lab-to-field gap.

Example (Colab/Kaggle GPU):
    python src/disease_train.py --data_dir /path/to/plantvillage/color \
        --arch efficientnet_b0 --epochs_head 3 --epochs_finetune 12

Quick CPU smoke test:
    python tools/make_synthetic_dataset.py --out /tmp/synth
    python src/disease_train.py --data_dir /tmp/synth/train --no_pretrained \
        --arch mobilenet_v3_large --img_size 96 --epochs_head 1 --epochs_finetune 1
"""

import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from disease_common import (
    ARCHS, build_model, eval_transform, get_device, train_transform,
)


# ------------------------------------------------------------------ data
class TransformSubset(Dataset):
    """A subset of an ImageFolder with its own transform (so train gets
    augmentation and val/test do not, without re-scanning the folder)."""

    def __init__(self, base: datasets.ImageFolder, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, y = self.base.samples[self.indices[i]]
        return self.transform(self.base.loader(path)), y


def make_splits(base, val_frac, test_frac, max_per_class, seed):
    targets = np.array(base.targets)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(targets))
    if max_per_class:  # handy for debugging on a small slice
        keep = []
        for c in np.unique(targets):
            ci = idx[targets == c]
            rng.shuffle(ci)
            keep.extend(ci[:max_per_class])
        idx = np.array(sorted(keep))
    hold = val_frac + test_frac
    train_idx, tmp = train_test_split(idx, test_size=hold, stratify=targets[idx], random_state=seed)
    val_idx, test_idx = train_test_split(
        tmp, test_size=test_frac / hold, stratify=targets[tmp], random_state=seed)
    return train_idx, val_idx, test_idx


def class_weights(targets, num_classes):
    """sqrt-inverse-frequency weights: helps rare classes without letting
    them dominate the loss the way full inverse-frequency weighting would."""
    counts = np.bincount(targets, minlength=num_classes).astype(float)
    w = (counts.sum() / (num_classes * np.maximum(counts, 1))) ** 0.5
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ------------------------------------------------------- train / evaluate
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    all_logits, all_y = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_y.append(y.cpu())
    logits, y = torch.cat(all_logits), torch.cat(all_y)
    preds = logits.argmax(1)
    return {
        "loss": total_loss / n,
        "acc": (preds == y).float().mean().item(),
        "macro_f1": f1_score(y.numpy(), preds.numpy(), average="macro"),
        "logits": logits, "y": y,
    }


def set_trainable(model, head, phase):
    for p in model.parameters():
        p.requires_grad = phase == "finetune"
    for p in head.parameters():
        p.requires_grad = True


def fit_temperature(logits, labels):
    """Single-parameter calibration: divide logits by T fitted on val NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(logits, labels, n_bins=15):
    probs = torch.softmax(logits, 1)
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return ece


def top_k_acc(logits, labels, k=3):
    k = min(k, logits.size(1))
    return (logits.topk(k, 1).indices == labels.unsqueeze(1)).any(1).float().mean().item()


# ------------------------------------------------------------- reporting
def save_confusion_matrix(y, preds, class_names, out_path):
    cm = confusion_matrix(y, preds, labels=range(len(class_names)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6, len(class_names) * 0.28)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Row-normalised confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    pairs = [(int(cm[i, j]), class_names[i], class_names[j])
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    return [{"count": c, "true": t, "predicted": p} for c, t, p in sorted(pairs, reverse=True)[:15]]


def _norm_name(s):
    tokens = re.sub(r"[^a-z0-9]+", " ", s.lower()).split()
    return frozenset(t for t in tokens if t not in {"leaf", "leaves"})


def evaluate_field(model, field_dir, class_names, field_map_path, img_size, device, batch_size, workers):
    """Score on real field photos (e.g. PlantDoc). Folder names rarely match
    PlantVillage's, so we map them: explicit JSON map first, then a
    token-matching heuristic. Unmapped classes are reported, not silently used."""
    base = datasets.ImageFolder(field_dir)
    explicit = json.load(open(field_map_path)) if field_map_path else {}
    train_norm = {_norm_name(c): i for i, c in enumerate(class_names)}
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    mapping, unmatched = {}, []
    for f_idx, f_name in enumerate(base.classes):
        if f_name in explicit and explicit[f_name] in name_to_idx:
            mapping[f_idx] = name_to_idx[explicit[f_name]]
        elif _norm_name(f_name) in train_norm:
            mapping[f_idx] = train_norm[_norm_name(f_name)]
        else:
            unmatched.append(f_name)
    keep = [i for i, (_, y) in enumerate(base.samples) if y in mapping]
    print(f"\n[field] {len(mapping)}/{len(base.classes)} classes mapped, {len(keep)} images used")
    if unmatched:
        print(f"[field] unmapped (skipped) -> supply --field_map to include: {unmatched}")
    if not keep:
        print("[field] nothing to evaluate")
        return None

    class Mapped(TransformSubset):
        def __getitem__(self, i):
            x, y = super().__getitem__(i)
            return x, mapping[y]

    loader = DataLoader(Mapped(base, keep, eval_transform(img_size)),
                        batch_size=batch_size, num_workers=workers)
    res = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    print(f"[field] accuracy={res['acc']:.4f}  macro_f1={res['macro_f1']:.4f}")
    return {"images": len(keep), "classes_mapped": len(mapping),
            "unmapped": unmatched, "acc": res["acc"], "macro_f1": res["macro_f1"]}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--out_dir", default="models/disease")
    ap.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_head", type=int, default=3)
    ap.add_argument("--epochs_finetune", type=int, default=12)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_finetune", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4, help="early-stop patience (fine-tune phase)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--max_per_class", type=int, default=0, help="debug: cap images per class")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_pretrained", action="store_true", help="random init (offline smoke tests)")
    ap.add_argument("--field_dir", default=None, help="optional real-field images for OOD evaluation")
    ap.add_argument("--field_map", default=None, help="optional JSON {field_folder: train_class}")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | AMP: {use_amp} | arch: {a.arch}")

    base = datasets.ImageFolder(a.data_dir)
    class_names = base.classes
    K = len(class_names)
    print(f"Found {len(base)} images in {K} classes")
    tr_i, va_i, te_i = make_splits(base, a.val_frac, a.test_frac, a.max_per_class, a.seed)
    print(f"Split -> train {len(tr_i)} | val {len(va_i)} | test {len(te_i)}")

    mk = lambda ds, shuffle: DataLoader(
        ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.num_workers,
        pin_memory=device.type == "cuda", drop_last=False)
    train_loader = mk(TransformSubset(base, tr_i, train_transform(a.img_size)), True)
    val_loader = mk(TransformSubset(base, va_i, eval_transform(a.img_size)), False)
    test_loader = mk(TransformSubset(base, te_i, eval_transform(a.img_size)), False)

    model, head = build_model(a.arch, K, pretrained=not a.no_pretrained)
    model.to(device)
    weights = class_weights(np.array(base.targets)[tr_i], K).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best = {"f1": -1.0, "state": None, "epoch": -1}
    history = []
    head_ids = {id(p) for p in head.parameters()}

    def run_phase(name, epochs, lr, patience=None):
        if epochs <= 0:
            return
        set_trainable(model, head, name)
        if name == "head":
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        else:  # fine-tune: backbone at lr, fresh head at 10x lr
            bb = [p for p in model.parameters() if id(p) not in head_ids]
            hd = [p for p in model.parameters() if id(p) in head_ids]
            opt = torch.optim.AdamW([{"params": bb, "lr": lr}, {"params": hd, "lr": lr * 10}],
                                    weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bad = 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_loader, criterion, device, opt, scaler, use_amp)
            va = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
            sched.step()
            history.append({"phase": name, "epoch": ep, "train_loss": tr["loss"], "train_acc": tr["acc"],
                            "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"]})
            flag = ""
            if va["macro_f1"] > best["f1"]:
                best.update(f1=va["macro_f1"], state=copy.deepcopy(model.state_dict()),
                            epoch=len(history))
                bad, flag = 0, " *best*"
            else:
                bad += 1
            print(f"[{name} {ep:02d}/{epochs}] train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} | "
                  f"{time.time() - t0:.0f}s{flag}")
            if patience and bad >= patience:
                print(f"Early stopping ({patience} epochs without val F1 improvement)")
                break

    run_phase("head", a.epochs_head, a.lr_head)
    run_phase("finetune", a.epochs_finetune, a.lr_finetune, patience=a.patience)

    model.load_state_dict(best["state"])
    print(f"\nLoaded best weights (history step {best['epoch']}, val macro-F1 {best['f1']:.4f})")

    # --- calibration on validation logits, then honest test evaluation
    val_res = run_epoch(model, val_loader, criterion, device)
    temperature = fit_temperature(val_res["logits"], val_res["y"])
    te = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    ece_raw = expected_calibration_error(te["logits"], te["y"])
    ece_cal = expected_calibration_error(te["logits"] / temperature, te["y"])
    preds = te["logits"].argmax(1).numpy()
    y = te["y"].numpy()
    report = classification_report(y, preds, labels=range(K), target_names=class_names,
                                   output_dict=True, zero_division=0)
    confusions = save_confusion_matrix(y, preds, class_names, out / "confusion_matrix.png")
    print(f"TEST  acc {te['acc']:.4f} | top-3 {top_k_acc(te['logits'], te['y']):.4f} | "
          f"macro-F1 {te['macro_f1']:.4f}")
    print(f"Calibration: T={temperature:.3f} | ECE {ece_raw:.4f} -> {ece_cal:.4f}")
    worst = sorted(((v["f1-score"], k) for k, v in report.items() if k in class_names))[:5]
    print("Weakest classes (F1):", [(k, round(f, 3)) for f, k in worst])

    field = None
    if a.field_dir:
        field = evaluate_field(model, a.field_dir, class_names, a.field_map, a.img_size,
                               device, a.batch_size, a.num_workers)
        if field:
            print(f"Lab->field gap: test acc {te['acc']:.3f} vs field acc {field['acc']:.3f}")

    # --- save artefacts
    metrics = {
        "arch": a.arch, "img_size": a.img_size, "num_classes": K,
        "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
        "best_val_macro_f1": best["f1"], "test_acc": te["acc"],
        "test_top3": top_k_acc(te["logits"], te["y"]), "test_macro_f1": te["macro_f1"],
        "temperature": temperature, "ece_uncalibrated": ece_raw, "ece_calibrated": ece_cal,
        "field_eval": field, "top_confusions": confusions,
        "pretrained": not a.no_pretrained,
    }
    torch.save({"arch": a.arch, "img_size": a.img_size, "class_names": class_names,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "temperature": temperature, "metrics": metrics}, out / "disease_model.pt")
    json.dump(class_names, open(out / "class_names.json", "w"), indent=2)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    json.dump(report, open(out / "classification_report.json", "w"), indent=2)
    json.dump(history, open(out / "history.json", "w"), indent=2)

    try:  # TorchScript for mobile/edge deployment
        model.eval().cpu()
        traced = torch.jit.trace(model, torch.randn(1, 3, a.img_size, a.img_size))
        traced.save(str(out / "disease_model.ts.pt"))
    except Exception as e:  # export is a convenience, never fail training over it
        print(f"(TorchScript export skipped: {e})")
    print(f"\nSaved everything to {out}/")


if __name__ == "__main__":
    main()

