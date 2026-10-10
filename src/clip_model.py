"""Shared OpenCLIP model configuration and loading helpers."""

from __future__ import annotations

from pathlib import Path
import os
from shutil import copy2


MODEL_ARCHITECTURE = "ViT-B-32"
PRETRAINED_WEIGHTS = "laion2b_s34b_b79k"
TEXT_PROMPT_TEMPLATE = "a photo of {}"
PORTABLE_WEIGHTS = Path("search") / "model" / "open_clip_model.safetensors"


def portable_weights_path(cache_dir: Path) -> Path:
    return cache_dir.parent / PORTABLE_WEIGHTS


def preserve_portable_weights(cache_dir: Path) -> Path:
    """Flatten the downloaded checkpoint into a USB/Windows-safe regular file."""
    destination = portable_weights_path(cache_dir)
    if destination.is_file() and not destination.is_symlink():
        return destination
    candidates = [
        path
        for path in cache_dir.rglob("open_clip_model.safetensors")
        if path.resolve() != destination.resolve()
    ]
    if not candidates:
        raise RuntimeError("Downloaded OpenCLIP checkpoint could not be located")
    source = candidates[0].resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, destination)
    return destination


def load_clip(cache_dir: Path | None = None):
    """Load the exact image/text model pair used by indexing and retrieval."""
    try:
        import open_clip
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Semantic-search dependencies are missing. See requirements-search.txt."
        ) from error

    # Avoid oversubscribing this modest CPU while leaving one core responsive.
    torch.set_num_threads(max(1, min(7, (os.cpu_count() or 2) - 1)))
    kwargs = {"device": "cpu"}
    pretrained = PRETRAINED_WEIGHTS
    if cache_dir is not None:
        local_weights = portable_weights_path(cache_dir)
        if local_weights.is_file():
            pretrained = str(local_weights)
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(cache_dir)
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ARCHITECTURE,
        pretrained=pretrained,
        **kwargs,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_ARCHITECTURE)
    preprocess_config = open_clip.get_model_preprocess_cfg(model)
    return model, preprocess, tokenizer, preprocess_config


def model_identity() -> dict[str, str]:
    import open_clip
    import torch

    return {
        "architecture": MODEL_ARCHITECTURE,
        "pretrained": PRETRAINED_WEIGHTS,
        "open_clip_version": open_clip.__version__,
        "torch_version": torch.__version__,
        "text_prompt_template": TEXT_PROMPT_TEMPLATE,
    }
