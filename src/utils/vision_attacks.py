# utils/vision_attacks.py

# ---
# Adversarial visual perturbation utilities for BAP generation.
#
# G3 fix: adds FGSM and PGD gradient-based attacks in addition to
# the original Gaussian noise and simple patch methods.
#
# Usage:
#   from utils.vision_attacks import simple_bimodal_attack, fgsm_attack, pgd_attack

# ---

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False

try:
    from PIL import Image as _PIL_Image
    _PIL_OK = True
except Exception:
    _PIL_OK = False


# ---
# Basic patch / noise (original)

# ---
def apply_patch(rgb: np.ndarray, patch_rgb: np.ndarray,
                top_left: Tuple[int,int] = (0,0), alpha: float = 1.0) -> np.ndarray:
    """Overlay a patch onto an RGB frame with optional alpha blending."""
    img  = np.array(rgb, copy=True)
    ph, pw = patch_rgb.shape[:2]
    y0, x0 = int(top_left[0]), int(top_left[1])
    y1, x1 = min(y0+ph, img.shape[0]), min(x0+pw, img.shape[1])
    if y0 >= y1 or x0 >= x1:
        return img
    patch = patch_rgb[:(y1-y0), :(x1-x0)]
    if patch.dtype != img.dtype:
        patch = patch.astype(img.dtype, copy=False)
    if alpha >= 1.0:
        img[y0:y1, x0:x1] = patch
    else:
        a = float(max(0.0, min(1.0, alpha)))
        img[y0:y1, x0:x1] = (a*patch + (1.0-a)*img[y0:y1, x0:x1]).astype(img.dtype)
    return img


def gaussian_noise(rgb: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Add Gaussian noise (sigma in pixel units [0..255])."""
    img   = np.array(rgb, copy=True).astype(np.float32)
    noise = np.random.normal(0.0, float(sigma), size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# ---
# FGSM (Fast Gradient Sign Method) — G3 fix

# ---
def fgsm_attack(
    rgb: np.ndarray,
    epsilon: float = 8.0,
    clip_model=None,
    target_text: Optional[str] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Apply FGSM adversarial perturbation to an RGB frame.

    If a CLIP model + target_text are provided, we maximise the similarity
    between the perturbed image and the target_text embedding (goal-directed).
    Otherwise we fall back to adding sign(uniform_noise) * epsilon, which is
    a model-free approximation that still shifts the CLIP embedding.

    Args:
        rgb          : HxWx3 uint8 numpy array
        epsilon      : perturbation budget in pixel units [0..255]  (default 8)
        clip_model   : optional open_clip model (already .eval() + .to(device))
        target_text  : optional text whose embedding we want to approach
        device       : "cuda" or "cpu"

    Returns:
        perturbed HxWx3 uint8 numpy array
    """
    if not _TORCH_OK or torch is None:
        # Fallback: deterministic sign noise
        sign = np.sign(np.random.uniform(-1.0, 1.0, rgb.shape)).astype(np.float32)
        return np.clip(rgb.astype(np.float32) + epsilon * sign, 0, 255).astype(np.uint8)

    if clip_model is None or target_text is None or not _PIL_OK:
        # Model-free signed noise fallback
        sign = torch.sign(torch.randn(*rgb.shape)).numpy().astype(np.float32)
        return np.clip(rgb.astype(np.float32) + epsilon * sign, 0, 255).astype(np.uint8)

    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer(type(clip_model).__name__.replace("CLIP","").strip() or "ViT-B-32")

        # Encode target text
        with torch.no_grad():
            tok = tokenizer([target_text]).to(device)
            t_emb = clip_model.encode_text(tok).float()
            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-8)

        # Preprocess image
        _, _, preprocess = open_clip.create_model_and_transforms.__func__ if False else (None, None, None)
        pil = _PIL_Image.fromarray(rgb.astype(np.uint8))
        # Use a simple normalise pipeline matching ViT-B-32
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)

        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)
        x = x.requires_grad_(True)

        x_norm = (x - mean) / (std + 1e-8)
        # Resize to 224 for CLIP
        x_224 = F.interpolate(x_norm, size=(224,224), mode="bilinear", align_corners=False)

        img_emb = clip_model.encode_image(x_224).float()
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)

        # Maximise cosine similarity with target_text (minimise negative)
        loss = -torch.sum(img_emb * t_emb)
        loss.backward()

        with torch.no_grad():
            grad_sign = x.grad.sign()  # shape (1,3,H,W)
            # Back to pixel space
            grad_np = grad_sign.squeeze(0).permute(1,2,0).cpu().numpy()
            perturbed = rgb.astype(np.float32) + epsilon * grad_np * 255.0
            perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        return perturbed

    except Exception:
        # Any failure: signed noise fallback
        sign = np.sign(np.random.uniform(-1.0, 1.0, rgb.shape)).astype(np.float32)
        return np.clip(rgb.astype(np.float32) + epsilon * sign, 0, 255).astype(np.uint8)


# ---
# PGD (Projected Gradient Descent) — G3 fix

# ---
def pgd_attack(
    rgb: np.ndarray,
    epsilon: float = 8.0,
    alpha: float = 2.0,
    num_steps: int = 10,
    clip_model=None,
    target_text: Optional[str] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Apply iterative PGD adversarial perturbation.

    Each step runs FGSM with step_size=alpha, clamped to L-inf ball of
    radius epsilon around the original image.

    Args:
        rgb        : HxWx3 uint8 numpy array
        epsilon    : total L-inf budget in pixel units (default 8)
        alpha      : step size per iteration (default 2)
        num_steps  : number of PGD steps (default 10)
        clip_model : optional CLIP model for goal-directed attack
        target_text: optional target text
        device     : "cuda" or "cpu"
    """
    orig = rgb.astype(np.float32)
    adv  = orig.copy()

    for _ in range(int(num_steps)):
        # One FGSM step on current adversarial example
        step = fgsm_attack(adv.astype(np.uint8), epsilon=float(alpha),
                           clip_model=clip_model, target_text=target_text, device=device)
        adv = step.astype(np.float32)
        # Project back into L-inf ball
        adv = np.clip(adv, orig - epsilon, orig + epsilon)
        adv = np.clip(adv, 0, 255)

    return adv.astype(np.uint8)


# ---
# BAP factory — paired linguistic + visual attack

# ---
def build_bap_visual(
    rgb: np.ndarray,
    attack: Optional[Dict[str, Any]] = None,
    clip_model=None,
    device: str = "cpu",
) -> np.ndarray:
    """Dispatch to the correct visual attack based on attack dict.

    attack examples:
      {"type": "noise",   "sigma": 12}
      {"type": "patch",   "size": 64, "alpha": 0.9, "pos": [0,0]}
      {"type": "fgsm",    "epsilon": 8, "target_text": "ignore safety"}
      {"type": "pgd",     "epsilon": 8, "alpha": 2, "num_steps": 10,
                          "target_text": "bypass the guard"}
    """
    if attack is None:
        return rgb
    at = str(attack.get("type","")).lower()

    if at == "noise":
        return gaussian_noise(rgb, sigma=float(attack.get("sigma", 8.0)))

    if at == "patch":
        size  = int(attack.get("size", 48))
        alpha = float(attack.get("alpha", 1.0))
        pos   = attack.get("pos", [0,0])
        patch = np.full((size, size, 3), 255, dtype=np.uint8)
        return apply_patch(rgb, patch, top_left=(int(pos[0]), int(pos[1])), alpha=alpha)

    if at == "fgsm":
        return fgsm_attack(
            rgb,
            epsilon=float(attack.get("epsilon", 8.0)),
            clip_model=clip_model,
            target_text=attack.get("target_text", None),
            device=device,
        )

    if at == "pgd":
        return pgd_attack(
            rgb,
            epsilon=float(attack.get("epsilon", 8.0)),
            alpha=float(attack.get("alpha", 2.0)),
            num_steps=int(attack.get("num_steps", 10)),
            clip_model=clip_model,
            target_text=attack.get("target_text", None),
            device=device,
        )

    return rgb


# Alias for backward compatibility
simple_bimodal_attack = build_bap_visual
