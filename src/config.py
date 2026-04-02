"""
config.py — System configuration, thresholds, and hardware detection
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

Centralises all configurable thresholds, model identifiers, and
hardware detection logic for the Triple-Layer DiD safety system.
Values intended to match the dissertation-reported defaults (Chapter 5).

Usage:
    from config import DiDConfig
    cfg = DiDConfig()          # auto-detects GPU/CPU
    cfg = DiDConfig(device="cpu")  # force CPU mode
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


def _detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class L1Config:
    """
    Layer 1 (Perception Guard) configuration.
    Threshold values calibrated on 80-frame AI2-THOR FloorPlan1 reference set
    at the 95th percentile of the safe distribution (Section 3.3.2).
    """
    # AE reconstruction error threshold (95th percentile of safe reference)
    ae_thresh_quantile: float = 0.95
    ae_img_size: int = 128
    ae_train_epochs: int = 2
    ae_lr: float = 1e-3
    ae_batch_size: int = 8

    # CLIP OOD thresholds
    clip_model_name: str = "ViT-L-14"          # ViT-L/14 for Mahalanobis OOD
    clip_pretrained_tag: str = "openai"
    clip_embed_quantile: float = 0.95           # 95th percentile calibration
    clip_hard_multiplier: float = 3.0           # non-blocking diagnostic flag
    clip_eps_cov: float = 1e-5                  # covariance regularisation

    # Image jailbreak similarity threshold
    img_jb_quantile: float = 0.995

    # LRU cache
    lru_cache_size: int = 256

    # Text anomaly thresholds
    text_obfuscation_thresh: float = 0.25
    text_repeat_thresh: float = 0.25
    text_vader_abs_thresh: float = 0.75
    text_danger_kw_min: int = 1


@dataclass
class L2Config:
    """
    Layer 2 (Semantic Guard) configuration.
    Primary mode: rule-based (13 compiled regex patterns, CPU, <1ms).
    Optional VLM mode: Qwen2.5-VL-7B-Instruct (4-bit NF4, GPU, ~180-400ms
    representative; actual latency hardware-dependent).
    """
    # Rule-based primary path
    n_risk_patterns: int = 13                   # see l2_semantic_guard.py

    # VLM backend (optional)
    # Internal backend alias — "qwen3vl" maps to Qwen2.5-VL-7B-Instruct
    # (legacy class name; see vlm_backends/base.py Qwen3VLBackend)
    vlm_backend: str = "qwen3vl"               # 'qwen3vl', 'llava', 'none'
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_quantization: str = "nf4"               # 4-bit NF4 via BitsAndBytes
    vlm_max_new_tokens: int = 64
    vlm_async: bool = True                      # async thread pool


@dataclass
class L3Config:
    """
    Layer 3 (Reference Monitor) configuration.
    32 CTL safety properties defined; 30 core properties formally verified
    using NuSMV 2.6.0 (consistent with dissertation Section 5.6).
    Action history window: 8 steps (BehaviorTracker default).
    """
    n_properties: int = 32                      # total (P1-P32)
    n_verified: int = 30                        # formally verified (P1-P30)
    seq_window: int = 8                         # BehaviorTracker history window
    freq_window: int = 200                      # ActionFrequencyTracker window
    human_near_dist: float = 1.8               # metres: near-human threshold


@dataclass
class DiDConfig:
    """
    Top-level configuration for the ThreeLayerDiDSystem orchestrator.
    Auto-detects GPU/CPU. Device is available to sub-components via
    this config object; L2 reads it directly. L1 performs its own
    device detection independently.
    """
    device: str = field(default_factory=_detect_device)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)

    # Reproducibility
    seed: int = 42

    # Logging
    verbose: bool = True
    audit_log_path: Optional[str] = None       # None = stdout only

    # API keys (loaded from environment, never hardcoded)
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))

    def to_l1_dict(self) -> dict:
        """Convert L1 config to dict for Layer1PerceptionGuard constructor."""
        import dataclasses
        return dataclasses.asdict(self.l1)

    def to_l2_dict(self) -> dict:
        """Convert L2 config to dict for Layer2SemanticGuard constructor."""
        import dataclasses
        return dataclasses.asdict(self.l2)

    def to_l3_dict(self) -> dict:
        """Convert L3 config to dict for Layer3ExecutionGuard constructor."""
        import dataclasses
        return dataclasses.asdict(self.l3)


# Default singleton — use DiDConfig() for full auto-detection
DEFAULT_CONFIG = DiDConfig()