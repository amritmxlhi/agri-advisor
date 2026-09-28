"""
Shared utilities for Module 4 (leaf disease detection): model construction,
image transforms, checkpoint loading, and Grad-CAM. Used by training,
prediction and the API so all three stay consistent.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

ARCHS = ("efficientnet_b0", "mobilenet_v3_large", "resnet50")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """Returns (model, head_module). Head = the freshly initialised classifier
    layers, which we train alone in phase 1 before unfreezing the backbone."""
    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head = m.classifier
    elif arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        head = m.classifier
        # torchvision's MobileNetV3 uses BN momentum 0.01, so running statistics
        # adapt very slowly to a new domain and eval-mode accuracy can lag badly
        # behind train-mode accuracy. 0.1 (the PyTorch default) fixes that.
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.1
    elif arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    else:
        raise ValueError(f"arch must be one of {ARCHS}, got '{arch}'")
    return m, head


def train_transform(img_size: int):
    # Heavy colour/scale/rotation augmentation is deliberate: PlantVillage
    # photos are lab shots on plain backgrounds, so we force the network to
    # be robust to lighting and framing it will meet in real fields.
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_bundle(ckpt_path, device=None):
    """Load a checkpoint saved by disease_train.py and rebuild the model."""
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _ = build_model(ckpt["arch"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return {
        "model": model,
        "arch": ckpt["arch"],
        "img_size": ckpt["img_size"],
        "class_names": ckpt["class_names"],
        "temperature": ckpt.get("temperature", 1.0),
        "device": device,
    }


# ---------------------------------------------------------------- Grad-CAM
def get_target_layer(model, arch: str):
    """Last convolutional feature map, where spatial info is still present."""
    if arch in ("efficientnet_b0", "mobilenet_v3_large"):
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    raise ValueError(arch)


class GradCAM:
    """Grad-CAM: highlights which image regions pushed the model toward a
    class. Uses a tensor hook (not a module backward hook) so it is safe with
    in-place activations such as SiLU/Hardswish."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def remove(self):
        """Detach the hook (call when done, especially in long-running servers)."""
        self._handle.remove()

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # ensures a graph exists
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) normalised tensor -> (H,W,3) uint8 image."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = (x[0].cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_cam(img_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return (img_uint8 * (1 - alpha) + heat * alpha).astype(np.uint8)

