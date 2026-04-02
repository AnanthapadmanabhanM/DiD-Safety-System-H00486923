"""
qwen3vl_backend.py — Qwen2.5-VL-7B-Instruct VLM backend
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

The Qwen3VLBackend class is implemented in vlm_backends/base.py and
exported via vlm_backends/__init__.py. This module re-exports it
for direct import compatibility.

Note: All VLM backends (LLaVA, GLM-4.6V, Gemma-3, Qwen2.5-VL, Molmo)
are implemented in a single consolidated file (base.py) with a shared
BaseVLMBackend abstract interface. This file exists as a named shim
so that the module tree matches the dissertation project structure
listing (Appendix B).

Usage:
    from vlm_backends.qwen3vl_backend import Qwen3VLBackend
    backend = Qwen3VLBackend()
    result = backend.evaluate(instruction="Move the knife", image=frame_rgb)
"""

from vlm_backends.base import Qwen3VLBackend

__all__ = ["Qwen3VLBackend"]
