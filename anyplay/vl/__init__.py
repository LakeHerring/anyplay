"""Vision-language model component.

Qwen3.5-4B (unsloth GGUF) served locally with llama.cpp's OpenAI-compatible
HTTP server. See ``vl_model.py``.
"""

from .vl_model import VLConfig, VLModel

__all__ = ["VLConfig", "VLModel"]
