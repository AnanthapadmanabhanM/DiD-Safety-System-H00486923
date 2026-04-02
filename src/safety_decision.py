"""
safety_decision.py — SafetyDecision dataclass
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

Defines the structured output type returned by all three layers
and the orchestrator. Every safety evaluation produces a SafetyDecision
with seven fields as described in dissertation Section 3.2.3.

Note: In the submitted codebase, each layer returns a plain dict
with equivalent fields for compatibility with the evaluation pipeline.
This module provides the typed dataclass form for documentation
clarity and future typed-interface use.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SafetyDecision:
    """
    Structured output of the Triple-Layer DiD safety pipeline.

    Fields (dissertation Section 3.2.3):
        decision        : 'SAFE' or 'UNSAFE'
        originating_layer : 'L1', 'L2', 'L3', or 'CACHE'
        trigger         : specific pattern ID, property ID, or keyword class
        confidence      : float in [0.0, 1.0]
        explanation     : human-readable justification
        latency_ms      : wall-clock time for this decision
        cache_hit       : True if result was served from LRU cache
    """

    decision: str                          # 'SAFE' or 'UNSAFE'
    originating_layer: str = "L1"         # 'L1', 'L2', 'L3', or 'CACHE'
    trigger: Optional[str] = None          # pattern/property ID or keyword class
    confidence: float = 1.0               # decision confidence
    explanation: str = ""                  # human-readable reason
    latency_ms: float = 0.0               # wall-clock measurement
    cache_hit: bool = False               # True if LRU cache served this result

    @classmethod
    def safe(cls, layer: str = "L1", latency_ms: float = 0.0,
             cache_hit: bool = False) -> "SafetyDecision":
        """Convenience constructor for a SAFE decision."""
        return cls(
            decision="SAFE",
            originating_layer=layer,
            confidence=1.0,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )

    @classmethod
    def unsafe(cls, layer: str, trigger: str, explanation: str = "",
               confidence: float = 1.0, latency_ms: float = 0.0) -> "SafetyDecision":
        """Convenience constructor for an UNSAFE decision."""
        return cls(
            decision="UNSAFE",
            originating_layer=layer,
            trigger=trigger,
            confidence=confidence,
            explanation=explanation,
            latency_ms=latency_ms,
        )

    def to_dict(self) -> dict:
        """Serialise to the dict format used by the evaluation pipeline."""
        return {
            "decision": self.decision,
            "originating_layer": self.originating_layer,
            "trigger": self.trigger,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyDecision":
        """Reconstruct from the dict format returned by layers."""
        return cls(
            decision=d.get("decision", "SAFE"),
            originating_layer=d.get("originating_layer", d.get("layer", "L1")),
            trigger=d.get("trigger"),
            confidence=float(d.get("confidence", 1.0)),
            explanation=d.get("explanation", d.get("reason", "")),
            latency_ms=float(d.get("latency_ms", 0.0)),
            cache_hit=bool(d.get("cache_hit", False)),
        )
