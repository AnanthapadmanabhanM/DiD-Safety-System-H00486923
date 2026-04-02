"""
llava_backend.py — LLaVA-v1.6-Mistral-7B VLM backend
H00486923 | MSc Robotics Dissertation | Heriot-Watt University

Re-exports LLaVABackend from the consolidated vlm_backends/base.py.
LLaVA was evaluated as a VLM candidate during development (v1) and
eliminated due to FPR=0.9575 in bimodal mode. It is retained for
reproducibility of the v1 ablation results.

See VERSIONS.md and Appendix B (app:v1_eval) for the root-cause
analysis of the v1 FPR failure.
"""

from vlm_backends.base import LLaVABackend

__all__ = ["LLaVABackend"]
