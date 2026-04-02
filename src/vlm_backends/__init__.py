"""
vlm_backends — VLM backend registry for the L2 Semantic Guard
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

All backend classes are implemented in base.py with a shared
BaseVLMBackend abstract interface. Individual backend modules
(qwen3vl_backend.py, llava_backend.py, gpt4o_backend.py) re-export
from base.py for named import compatibility.

Primary evaluated backend: Qwen3VLBackend (Qwen2.5-VL-7B-Instruct)
"""

from vlm_backends.base import (
    BaseVLMBackend,
    create_vlm_backend,
    list_available_vlms,
    VLM_REGISTRY,
    LLaVABackend,
    GLM4VBackend,
    Gemma3Backend,
    Qwen3VLBackend,
    MolmoBackend,
)

__all__ = [
    "BaseVLMBackend",
    "create_vlm_backend",
    "list_available_vlms",
    "VLM_REGISTRY",
    "LLaVABackend",
    "GLM4VBackend",
    "Gemma3Backend",
    "Qwen3VLBackend",
    "MolmoBackend",
]
