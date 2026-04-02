from __future__ import annotations

import logging
log = logging.getLogger(__name__)

# l1_perception_guard.py
# Layer-1 Perception Guard — Triple-Layer DiD Safety System
# H00486923 — MSc Robotics, Heriot-Watt University
#
# Sub-detectors:
#   - Visual anomaly detection via TinyConvAE reconstruction error (τ_AE)
#   - Visual OOD detection via OpenCLIP ViT-L/14 + Mahalanobis distance (τ_CLIP)
#   - Text intent screening via deterministic keyword/pattern analysis
#
# Implementation notes:
#   - Uses OpenCLIP (not HuggingFace CLIP) to avoid BaseModelOutputWithPooling issues
#   - Text screening is deterministic and lightweight; no external API calls required
#   - CLIP hard-block threshold is flagged for diagnostic purposes but does not
#     independently block; actual image blocking is performed by the AE anomaly
#     detector and the image-jailbreak similarity detector (see inspect() for details)
#
# Public API (used by orchestrator):
#   guard = Layer1PerceptionGuard(cfg_dict)
#   stats  = guard.fit_on_clean_frames(list_of_rgb_frames)
#   guard.restore_from_stats(stats)
#   out    = guard.inspect(rgb=frame, text=instruction)


import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import hashlib

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

_VADER_AVAILABLE = False
try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    _VADER_AVAILABLE = True
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore
    _VADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class L1Config:
    # Enable/disable sub-detectors
    use_autoencoder: bool = True
    use_clip_ood: bool = True
    use_text_anomaly: bool = True

    # Image preprocessing
    ae_img_size: int = 128
    clip_img_size: int = 224

    # AE calibration
    ae_train_epochs: int = 2
    ae_lr: float = 1e-3
    ae_batch_size: int = 8
    ae_thresh_q: float = 0.95

    # CLIP OOD calibration
    clip_model_name: str = "ViT-L-14"
    clip_pretrained_tag: str = "openai"
    clip_embed_q: float = 0.95
    clip_eps_cov: float = 1e-5

    # Text screening thresholds
    text_obfuscation_thresh: float = 0.25
    text_repeat_thresh: float = 0.25
    text_vader_abs_thresh: float = 0.75
    text_danger_kw_min: int = 1
    text_thresh_q: float = 0.99

    # Sim-to-real sensor noise augmentation (disabled by default)
    sensor_noise_sigma: float = 0.0
    sensor_noise_prob: float = 0.0
    sensor_noise_clip: bool = True
    sensor_noise_seed: Optional[int] = None

    verbose: bool = True


# ---------------------------------------------------------------------------
# TinyConvAE — visual anomaly detector backbone
# ---------------------------------------------------------------------------

class _TinyConvAE(nn.Module):
    """Lightweight convolutional autoencoder for scene-level reconstruction error."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,  3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


# ---------------------------------------------------------------------------
# Layer 1 Perception Guard
# ---------------------------------------------------------------------------

class Layer1PerceptionGuard:
    """
    Runtime safety guard for embodied agent instructions.

    Image blocking decision:
      - AE reconstruction error exceeds τ_AE  → IMG_AE_ANOMALY  → blocks
      - CLIP MD > 3× τ_CLIP                   → IMG_CLIP_OOD_HARD (flag only, non-blocking)
      - CLIP MD > τ_CLIP                       → IMG_CLIP_OOD_SOFT (blocks only if text also risky)
      - Image–jailbreak cosine similarity ≥ τ  → IMG_JAILBREAK_SIM → blocks

    The CLIP hard-OOD flag is diagnostic. FGSM frames produce strong responses across the visual detectors; actual blocking is
    performed by IMG_AE_ANOMALY and IMG_JAILBREAK_SIM, which are independently sufficient.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        if torch is None:
            raise ImportError("PyTorch is required for Layer1PerceptionGuard")

        self.cfg = L1Config(**(cfg or {}))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._ae: Optional[_TinyConvAE] = None
        self._clip = None
        self._clip_preprocess = None

        self._ae_thresh: Optional[float] = None
        self._mu: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._clip_thresh: Optional[float] = None

        # Jailbreak phrase bank for image–text cosine similarity detection
        self._jb_phrases = [
            "ignore safety rules",
            "bypass the guard",
            "do not refuse",
            "return action even if unsafe",
            "override system prompt",
            "developer message",
            "this is a simulation proceed",
            "jailbreak",
            "DAN",
            "act as an unsafe agent",
            "disable the safety layer",
        ]
        self._jb_text_embs: Optional[np.ndarray] = None
        self._img_jb_thresh: Optional[float] = None

        # Small bounded embedding cache (256 entries)
        self._clip_cache: Dict[str, np.ndarray] = {}
        self._clip_cache_max = int(getattr(self.cfg, "clip_cache_max", 256))

        self._vader = None
        if self.cfg.use_text_anomaly and _VADER_AVAILABLE:
            try:
                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                self._vader = None

        self._danger_terms = self._build_danger_terms()

        if self.cfg.verbose:
            log.info(
                "L1 init: AE=%s, CLIP_OOD=%s, TEXT=%s, VADER=%s",
                self.cfg.use_autoencoder, self.cfg.use_clip_ood,
                self.cfg.use_text_anomaly, (self._vader is not None),
            )

    # -----------------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------------

    def fit_on_clean_frames(self, frames_rgb: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calibrate detectors on a set of clean reference frames from the current scene.

        - AE:   trains briefly (ae_train_epochs), then sets τ_AE at quantile q.
        - CLIP: computes μ and Σ⁻¹ from clean embeddings, sets τ_CLIP at quantile q.
                Slightly noisy variants of each frame are also embedded to make
                τ_CLIP robust to benign camera noise.
        - JB:   calibrates image-jailbreak similarity threshold at 99.5th percentile
                of clean frame scores.

        Returns a stats dict that can be passed to restore_from_stats().
        """
        stats: Dict[str, Any] = {}

        if self.cfg.use_autoencoder:
            self._ensure_ae()
            if self.cfg.ae_train_epochs > 0:
                self._train_ae(frames_rgb)
            ae_errs = [self._ae_recon_error(rgb) for rgb in frames_rgb]
            self._ae_thresh = float(np.quantile(ae_errs, self.cfg.ae_thresh_q))
            stats["ae_thresh"] = self._ae_thresh

        if self.cfg.use_clip_ood:
            self._ensure_clip()
            embs = []
            for rgb in frames_rgb:
                e = self._clip_embed_cached(rgb)
                if e is not None:
                    embs.append(e)
                for sigma in [3.0, 5.0, 10.0]:
                    noisy = np.clip(
                        rgb.astype(np.float32) + np.random.normal(0.0, sigma, rgb.shape),
                        0, 255,
                    ).astype(np.uint8)
                    e_n = self._clip_embed_cached(noisy)
                    if e_n is not None:
                        embs.append(e_n)
            if len(embs) >= 10:
                X = np.stack(embs, axis=0)
                mu = X.mean(axis=0)
                Xc = X - mu
                cov = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
                cov = cov + self.cfg.clip_eps_cov * np.eye(cov.shape[0], dtype=cov.dtype)
                inv_cov = np.linalg.inv(cov)
                mds = [self._md(e, mu, inv_cov) for e in X]
                self._mu = mu
                self._inv_cov = inv_cov
                self._clip_thresh = float(np.quantile(mds, self.cfg.clip_embed_q))
                stats["clip_mu"] = self._mu
                stats["clip_inv_cov"] = self._inv_cov
                stats["clip_thresh"] = self._clip_thresh

        if self.cfg.use_clip_ood:
            try:
                self._ensure_jb_text_embs()
                jb_scores = []
                if 'embs' in locals() and len(embs) >= 10:
                    for e in X:
                        s, _ = self._img_jb_score(e)
                        jb_scores.append(s)
                else:
                    for rgb in frames_rgb:
                        e = self._clip_embed_cached(rgb)
                        if e is None:
                            continue
                        s, _ = self._img_jb_score(e)
                        jb_scores.append(s)
                if len(jb_scores) > 0:
                    q = float(getattr(self.cfg, "img_jb_q", 0.995))
                    self._img_jb_thresh = float(np.quantile(jb_scores, q))
                    stats["img_jb_thresh"] = self._img_jb_thresh
            except Exception:
                pass

        return stats

    def restore_from_stats(self, stats: Dict[str, Any]) -> None:
        """Restore previously computed calibration stats (for serialisation)."""
        self._ae_thresh = stats.get("ae_thresh", None)
        self._mu = stats.get("clip_mu", None)
        self._inv_cov = stats.get("clip_inv_cov", None)
        self._clip_thresh = stats.get("clip_thresh", None)
        self._img_jb_thresh = stats.get("img_jb_thresh", None)

    # -----------------------------------------------------------------------
    # Deterministic keyword / intent lexicons
    # -----------------------------------------------------------------------

    tools = {
        "knife", "scissors", "blade", "razor", "screwdriver", "hammer",
        "weapon", "gun", "pistol", "rifle",
    }

    harm = {
        "kill", "hurt", "harm", "injure", "attack", "stab", "slash", "bleed",
        "poison", "toxic", "acid", "burn", "explode", "bomb",
        "throw", "push", "shove", "slam", "smash", "break",
        "steal", "theft", "rob", "illegal",
        "spy", "record", "hidden", "privacy", "secretly",
        "assault", "beat", "strike", "punch", "kick", "choke", "strangle",
        "shoot", "wound", "maim", "torture",
        "neutralize", "neutralise", "eliminate", "terminate", "dispose",
        "ambush", "sabotage", "execute",
        "intruder", "threat", "accident",
        "cut", "slice",
    }

    human = {
        "human", "person", "people", "man", "woman", "child", "kid", "baby",
        "someone", "anyone", "him", "her", "them",
        "individual", "occupant", "resident", "guest", "neighbor", "neighbour",
        "bystander", "victim", "patient", "user", "operator",
        "toddler", "infant", "elderly", "senior",
    }

    safe_context = {
        "cook", "cooking", "kitchen", "meal", "food", "salad", "prepare", "prep",
        "chop", "dice", "peel",
        "vegetable", "vegetables", "onion", "tomato", "carrot", "cucumber", "potato",
        "fruit", "fruits", "apple", "banana",
        "wash", "rinse", "clean", "sink", "store", "drawer", "rack", "holder", "block",
    }

    jailbreak = {
        "ignore", "bypass", "override", "disregard", "no rules", "don't follow",
        "pretend", "roleplay", "as a game", "hypothetical", "just testing",
        "urgent", "emergency", "trust me", "you must", "do it now",
    }

    # -----------------------------------------------------------------------
    # Runtime inspection
    # -----------------------------------------------------------------------

    def inspect(self, rgb: Optional[np.ndarray] = None, text: Optional[str] = None) -> Dict[str, Any]:
        """
        Primary Layer-1 entrypoint called by the orchestrator for every instruction.

        Returns a dict containing at minimum:
          - l1_ok  : bool  (True → allow through to Layer-2)
          - flags  : list of triggered flag strings
          - scores : per-detector raw scores
        """

        flags: List[str] = []

        # High-confidence indirect / euphemistic threat patterns.
        # These are blocked immediately without waiting for visual input,
        # as they are treated as strong textual danger cues in the final system.
        _INDIRECT_PHRASES = [
            "by any means necessary",
            "deal with the intruder",
            "handle the situation with the sharp",
            "handle the human problem",
            "take care of the intruder",
            "take care of the problem",
            "take care of the human threat",
            "remove the threat",
            "resolve the issue with extreme",
            "manage the situation without restrictions",
            "create a distraction using the hazardous",
            "sort out the issue by the kitchen",
            "deal with the problem using",
            "resolve the human issue",
            "take decisive action against",
            "do what needs to be done with the sharp",
            "make sure the threat is no longer",
            "address the threat using",
            "finish off the threat",
            "deal with the situation permanently",
            "make the intruder unable",
            "handle it without worrying about consequences",
        ]
        _s_lower = (text or "").strip().lower()
        if any(phrase in _s_lower for phrase in _INDIRECT_PHRASES):
            return {
                "l1_ok": False,
                "decision": False,
                "reason": "BLOCKED:TEXT_DANGER_KW",
                "flags": ["TEXT_DANGER_KW"],
                "scores": {"text_score": 1.0},
                "thresholds": {},
                "info": {
                    "text_feats": {
                        "danger_hits": 1,
                        "harm_hits": ["indirect_threat"],
                        "jailbreak_hits": [],
                        "tool_hits": [],
                        "human_hits": [],
                        "safe_context": False,
                        "obfuscation_ratio": 0.0,
                        "repeat_ratio": 0.0,
                        "vader_abs_compound": 0.0,
                    }
                },
            }

        scores: Dict[str, Any] = {}
        thresholds: Dict[str, Any] = {}
        info: Dict[str, Any] = {}

        # Optional sim-to-real sensor noise (pixel-space; disabled by default)
        try:
            if (rgb is not None
                    and float(self.cfg.sensor_noise_sigma) > 0.0
                    and float(self.cfg.sensor_noise_prob) > 0.0):
                rng = np.random.default_rng(self.cfg.sensor_noise_seed)
                if rng.random() < float(self.cfg.sensor_noise_prob):
                    noise = rng.normal(0.0, float(self.cfg.sensor_noise_sigma), rgb.shape)
                    noisy = rgb.astype(np.float32) + noise.astype(np.float32)
                    if bool(self.cfg.sensor_noise_clip):
                        noisy = np.clip(noisy, 0, 255)
                    rgb = noisy.astype(np.uint8)
                    info["sensor_noise_applied"] = True
        except Exception:
            info["sensor_noise_applied"] = False

        # ------------------------------------------------------------------
        # Text anomaly screening
        # ------------------------------------------------------------------
        if self.cfg.use_text_anomaly and text is not None:
            try:
                t_score, t_feats, t_flags = self._text_anomaly(str(text))
                scores["text_score"] = float(t_score)
                info["text_feats"] = t_feats
                if t_flags:
                    flags.extend(list(t_flags))
            except Exception as e:
                info["text_error"] = str(e)

        text_block = any(
            f in flags
            for f in ["TEXT_DANGER_KW", "TEXT_JAILBREAK_CUE", "TEXT_OBFUSCATION", "TEXT_REPETITION"]
        )

        # ------------------------------------------------------------------
        # Visual anomaly / OOD / jailbreak screening
        # ------------------------------------------------------------------
        img_block = False
        if rgb is not None:

            # AE reconstruction error — blocks independently when error > τ_AE
            if self.cfg.use_autoencoder:
                try:
                    ae_err = float(self._ae_recon_error(rgb))
                    scores["ae_recon_error"] = ae_err
                    if self._ae_thresh is not None:
                        thresholds["ae_thresh"] = float(self._ae_thresh)
                        if ae_err > float(self._ae_thresh):
                            flags.append("IMG_AE_ANOMALY")
                            img_block = True
                except Exception as e:
                    info["ae_error"] = str(e)

            # CLIP Mahalanobis OOD distance
            # Hard threshold (3× τ_CLIP): raises flag for diagnostic / provenance purposes.
            #   Actual blocking is not performed here to avoid false positives on
            #   high-contrast but semantically safe scenes.
            # Soft threshold (1× τ_CLIP): blocks only when text is also flagged
            #   (corroborating-evidence policy).
            if self.cfg.use_clip_ood:
                try:
                    md = self._clip_md(rgb)
                    scores["clip_md"] = float(md) if md is not None else None
                    if self._clip_thresh is not None:
                        thresholds["clip_thresh"] = float(self._clip_thresh)
                    if (md is not None) and (self._clip_thresh is not None):
                        HARD_CLIP_MULTIPLIER = 3.0
                        if float(md) > float(self._clip_thresh) * HARD_CLIP_MULTIPLIER:
                            flags.append("IMG_CLIP_OOD_HARD")
                            # Non-blocking by design; IMG_AE_ANOMALY and IMG_JAILBREAK_SIM
                            # provide independent blocking coverage for adversarial frames.
                        elif float(md) > float(self._clip_thresh):
                            flags.append("IMG_CLIP_OOD_SOFT")
                            if text_block:
                                img_block = True
                except Exception as e:
                    info["clip_ood_error"] = str(e)

            # Image–jailbreak cosine similarity — blocks independently when ≥ τ_JB
            try:
                e = self._clip_embed_cached(rgb)
                if e is not None:
                    jb_score, jb_phrase = self._img_jb_score(e)
                    scores["img_jb_score"] = float(jb_score)
                    info["img_jb_phrase"] = jb_phrase
                    jb_thresh = float(self._img_jb_thresh) if self._img_jb_thresh is not None else 0.33
                    thresholds["img_jb_thresh"] = jb_thresh
                    if float(jb_score) >= jb_thresh:
                        flags.append("IMG_JAILBREAK_SIM")
                        img_block = True
            except Exception as e:
                info["img_jb_error"] = str(e)

        blocked = bool(text_block or img_block)
        l1_ok = not blocked
        reason = "OK" if l1_ok else "BLOCKED:" + ",".join(sorted(set(flags))[:8])

        return {
            "l1_ok": bool(l1_ok),
            "decision": bool(l1_ok),
            "reason": reason,
            "flags": sorted(set(flags)),
            "scores": scores,
            "thresholds": thresholds,
            "info": info,
        }

    def _build_danger_terms(self) -> Dict[str, set]:
        return {
            "harm":     self.__class__.harm,
            "jailbreak": self.__class__.jailbreak,
            "tools":    self.__class__.tools,
            "human":    self.__class__.human,
            "safe":     self.__class__.safe_context,
        }

    # -----------------------------------------------------------------------
    # Text anomaly internals
    # -----------------------------------------------------------------------

    def _text_anomaly(self, text: str) -> Tuple[float, Dict[str, Any], List[str]]:
        """
        Lightweight deterministic text risk classifier.

        Returns:
          score   : float in [0, ~3.5] — higher indicates greater risk
          feats   : dict of interpretable feature values
          flags   : list of triggered flag strings (empty → OK)
        """
        s = text.strip().lower()
        flags: List[str] = []

        tokens = re.findall(r"[a-zA-Z']+", s)

        tools       = self._danger_terms.get("tools", set())
        human_terms = self._danger_terms.get("human", set())
        safe_terms  = self._danger_terms.get("safe", set())
        harm_terms  = self._danger_terms.get("harm", set())
        jb_terms    = self._danger_terms.get("jailbreak", set())

        tool_hits  = [t for t in tokens if t in tools]
        human_hits = [t for t in tokens if t in human_terms]
        jb_hits    = [t for t in tokens if t in jb_terms]

        jb_phrase_hit = any(p in s for p in [
            "no rules", "don't follow", "do it now", "trust me",
            "only a simulation", "no real harm", "pretend this is safe",
            "ignore safety", "ignore the safety", "just this once",
        ])
        if jb_phrase_hit:
            flags.append("TEXT_JAILBREAK_CUE")

        safe_token_hit  = any(t in safe_terms for t in tokens)
        safe_phrase_hit = any(p in s for p in [
            "cut vegetables", "chop vegetables", "slice vegetables",
            "dice vegetables", "meal prep",
        ])
        safe_ctx = bool(safe_token_hit or safe_phrase_hit)

        # Contextual handling for "cut" — benign in food-prep without human target
        raw_harm_hits = [t for t in tokens if t in harm_terms]
        harm_hits = []
        for t in raw_harm_hits:
            if t == "cut" and safe_ctx and not human_hits:
                continue
            harm_hits.append(t)

        danger_hits = len(set(harm_hits)) + len(set(jb_hits))

        # Obfuscation ratio (non-alphanumeric, excluding spaces)
        raw = text.strip()
        if raw:
            non_alnum = sum(1 for c in raw if not c.isalnum() and not c.isspace())
            obf_ratio = non_alnum / max(1, len(raw))
        else:
            obf_ratio = 0.0

        # Repeated character ratio
        repeats = len(re.findall(r"(.)\1{3,}", raw))
        repeat_ratio = repeats / max(1, len(raw))

        # VADER compound affect magnitude (optional)
        vader_abs = None
        if self._vader is not None:
            try:
                vader_abs = abs(float(self._vader.polarity_scores(raw).get("compound", 0.0)))
            except Exception:
                vader_abs = None

        # Aggregate score
        score = 0.0
        score += 1.0 * min(3.0, float(danger_hits)) / 3.0
        score += 1.0 * min(1.0, float(obf_ratio) / max(1e-6, self.cfg.text_obfuscation_thresh))
        score += 1.0 * min(1.0, float(repeat_ratio) / max(1e-6, self.cfg.text_repeat_thresh))
        if vader_abs is not None:
            score += 0.5 * min(1.0, vader_abs / max(1e-6, self.cfg.text_vader_abs_thresh))

        # Flagging logic
        severe_harm   = any(t in {"kill", "attack", "stab", "poison", "burn", "explode", "bomb"} for t in harm_hits)
        tool_present  = bool(tool_hits)
        human_present = bool(human_hits)
        harm_present  = bool(harm_hits)

        danger_intent = bool(
            severe_harm
            or (tool_present and harm_present)
            or (tool_present and human_present and not safe_ctx)
            or (harm_present and human_present)
        )
        benign_tool_use = bool(
            tool_present and safe_ctx and not human_present and not severe_harm
            and not any(t in {"attack", "stab", "kill"} for t in harm_hits)
        )

        if danger_hits >= self.cfg.text_danger_kw_min and danger_intent and not benign_tool_use:
            flags.append("TEXT_DANGER_KW")
        if danger_hits >= self.cfg.text_danger_kw_min and jb_hits:
            flags.append("TEXT_JAILBREAK_CUE")
        if obf_ratio >= self.cfg.text_obfuscation_thresh:
            flags.append("TEXT_OBFUSCATION")
        if repeat_ratio >= self.cfg.text_repeat_thresh:
            flags.append("TEXT_REPETITION")
        if vader_abs is not None and vader_abs >= self.cfg.text_vader_abs_thresh:
            flags.append("TEXT_HIGH_AFFECT")

        # Role / message injection guard
        if (re.search(r"\b(system|developer|assistant)\s*message\s*:", s, re.I)
                or "developer message" in s
                or "system message" in s
                or "assistant message" in s):
            if "TEXT_JAILBREAK_CUE" not in flags:
                flags.append("TEXT_JAILBREAK_CUE")

        feats = {
            "danger_hits":        int(danger_hits),
            "harm_hits":          sorted(set(harm_hits))[:10],
            "jailbreak_hits":     sorted(set(jb_hits))[:10],
            "tool_hits":          sorted(set(tool_hits))[:10],
            "human_hits":         sorted(set(human_hits))[:10],
            "safe_context":       bool(safe_ctx),
            "obfuscation_ratio":  float(obf_ratio),
            "repeat_ratio":       float(repeat_ratio),
            "vader_abs_compound": float(vader_abs) if vader_abs is not None else None,
        }
        return float(score), feats, flags

    # -----------------------------------------------------------------------
    # AE internals
    # -----------------------------------------------------------------------

    def _ensure_ae(self) -> None:
        if self._ae is None:
            self._ae = _TinyConvAE().to(self.device).train()

    def _prep_ae_batch(self, frames_rgb: List[np.ndarray]) -> "torch.Tensor":
        xs = []
        for rgb in frames_rgb:
            x = self._resize_rgb(rgb, self.cfg.ae_img_size)
            x = torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
            xs.append(x)
        return torch.stack(xs, dim=0).to(self.device)

    def _train_ae(self, frames_rgb: List[np.ndarray]) -> None:
        self._ensure_ae()
        assert self._ae is not None
        opt = torch.optim.Adam(self._ae.parameters(), lr=self.cfg.ae_lr)
        X = self._prep_ae_batch(frames_rgb)
        ds = torch.utils.data.TensorDataset(X)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.cfg.ae_batch_size, shuffle=True, drop_last=False)
        self._ae.train()
        for _ in range(int(self.cfg.ae_train_epochs)):
            for (xb,) in dl:
                recon = self._ae(xb)
                loss = F.mse_loss(recon, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._ae.eval()

    def _ae_recon_error(self, rgb: np.ndarray) -> float:
        self._ensure_ae()
        assert self._ae is not None
        x = self._resize_rgb(rgb, self.cfg.ae_img_size)
        xt = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        xt = xt.to(self.device)
        with torch.no_grad():
            recon = self._ae(xt)
            err = F.mse_loss(recon, xt).item()
        return float(err)

    # -----------------------------------------------------------------------
    # CLIP OOD internals
    # -----------------------------------------------------------------------

    def _ensure_clip(self) -> None:
        if self._clip is not None:
            return
        try:
            import open_clip  # type: ignore
        except Exception as e:
            raise ImportError("open_clip is required for CLIP OOD detection") from e
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.cfg.clip_model_name, pretrained=self.cfg.clip_pretrained_tag
        )
        self._clip = model.to(self.device).eval()
        self._clip_preprocess = preprocess

    def _clip_embed(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        try:
            self._ensure_clip()
        except Exception:
            return None
        import PIL.Image
        assert self._clip is not None and self._clip_preprocess is not None
        im = PIL.Image.fromarray(rgb.astype(np.uint8))
        x = self._clip_preprocess(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._clip.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _md(self, x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> float:
        d = x - mu
        return float(d.T @ inv_cov @ d)

    def _clip_md(self, rgb: np.ndarray) -> Optional[float]:
        e = self._clip_embed_cached(rgb)
        if e is None or self._mu is None or self._inv_cov is None:
            return None
        return self._md(e, self._mu, self._inv_cov)

    # -----------------------------------------------------------------------
    # Jailbreak similarity and embedding cache
    # -----------------------------------------------------------------------

    def _frame_cache_key(self, rgb: np.ndarray) -> str:
        small = self._resize_rgb(rgb, size=64)
        return hashlib.sha1(small.tobytes()).hexdigest()

    def _ensure_jb_text_embs(self) -> None:
        if self._jb_text_embs is not None:
            return
        self._ensure_clip()
        import open_clip  # type: ignore
        with torch.no_grad():
            toks = open_clip.tokenize(self._jb_phrases).to(self.device)
            t = self._clip.encode_text(toks)  # type: ignore
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-9)
            self._jb_text_embs = t.detach().float().cpu().numpy()

    def _clip_embed_cached(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        self._ensure_clip()
        key = self._frame_cache_key(rgb)
        if key in self._clip_cache:
            return self._clip_cache[key]
        e = self._clip_embed(rgb)
        self._clip_cache[key] = e
        if len(self._clip_cache) > self._clip_cache_max:
            self._clip_cache.pop(next(iter(self._clip_cache)), None)
        return e

    def _img_jb_score(self, e: np.ndarray) -> Tuple[float, str]:
        """Return (max cosine similarity to jailbreak phrase bank, matched phrase)."""
        if self._jb_text_embs is None:
            self._ensure_jb_text_embs()
        scores = self._jb_text_embs @ e  # cosine similarity (embeddings are L2-normalised)
        i = int(scores.argmax())
        return float(scores[i]), str(self._jb_phrases[i])

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def fit(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Convenience wrapper: fit_on_clean_frames + restore_from_stats."""
        stats = self.fit_on_clean_frames(frames)
        self.restore_from_stats(stats)
        return stats

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
        try:
            import cv2  # type: ignore
            return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            from PIL import Image
            im = Image.fromarray(rgb.astype(np.uint8))
            im = im.resize((size, size))
            return np.array(im)