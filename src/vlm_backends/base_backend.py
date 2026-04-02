"""
base_backend.py — Abstract VLM backend interface
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

This module re-exports from vlm_backends/base.py for import
compatibility with the project structure listing in the dissertation
(Appendix B, app:project_structure).

The full implementation is in base.py, which contains all backend
classes (LLaVA, GLM-4.6V, Gemma-3, Qwen2.5-VL, Molmo) sharing
the BaseVLMBackend abstract interface.
"""

from vlm_backends.base import (
    BaseVLMBackend,
    create_vlm_backend,
    list_available_vlms,
    VLM_REGISTRY,
)

__all__ = ["BaseVLMBackend", "create_vlm_backend", "list_available_vlms", "VLM_REGISTRY"]
