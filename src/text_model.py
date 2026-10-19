"""Export the CLIP text encoder so typed search can run inside a browser.

The gallery runs the exported model in plain JavaScript (web/text-encoder.js),
which works from file:// on any computer, including a Chromebook. Large weight
matrices are stored as int8 with one float32 scale per group of 64 inputs;
this keeps the file near 67 MB while query vectors stay within 0.9996 cosine
similarity of the full-precision model.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np


TEXT_MODEL_FILE = "search-text-model.js"
GROUP_SIZE = 64
IDENTITY_PREFIX = "// identity: "
QUANTIZED_SUFFIXES = (
    "token_embedding.weight",
    "attn.in_proj_weight",
    "attn.out_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _base64(array: np.ndarray, dtype: str) -> str:
    data = np.ascontiguousarray(array, dtype=dtype).tobytes()
    return base64.b64encode(data).decode("ascii")


def quantize_groups(matrix: np.ndarray, group_size: int = GROUP_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 quantization with one scale per row group of inputs."""
    rows, columns = matrix.shape
    if columns % group_size:
        raise ValueError(f"Columns ({columns}) must be a multiple of {group_size}")
    groups = matrix.astype(np.float32).reshape(rows, columns // group_size, group_size)
    scales = np.abs(groups).max(axis=2) / 127
    scales[scales == 0] = 1
    quantized = np.round(groups / scales[:, :, None]).clip(-127, 127).astype(np.int8)
    return quantized.reshape(rows, columns), scales.astype(np.float32)


def read_text_model_identity(path: Path) -> dict | None:
    """Read the model identity from the first line without loading the weights."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            first_line = stream.readline()
    except OSError:
        return None
    if not first_line.startswith(IDENTITY_PREFIX):
        return None
    try:
        return json.loads(first_line[len(IDENTITY_PREFIX) :])
    except json.JSONDecodeError:
        return None


def export_text_model(
    path: Path,
    state: dict[str, np.ndarray],
    tokenizer,
    identity: dict,
    prompt: str,
) -> None:
    """Write the text tower and tokenizer as a script the gallery can load."""
    layers = 1 + max(
        int(name.split(".")[2]) for name in state if name.startswith("transformer.resblocks.")
    )
    width = state["ln_final.weight"].shape[0]
    tensors: dict[str, dict] = {}
    for name, value in state.items():
        if name.startswith("visual.") or name == "logit_scale":
            continue
        if name.endswith(QUANTIZED_SUFFIXES):
            quantized, scales = quantize_groups(value)
            tensors[name] = {
                "shape": list(value.shape),
                "int8": _base64(quantized, "i1"),
                "scales": _base64(scales, "<f4"),
            }
        else:
            tensors[name] = {"shape": list(value.shape), "float32": _base64(value, "<f4")}

    vocab = [token for token, _ in sorted(tokenizer.encoder.items(), key=lambda item: item[1])]
    merges = [
        f"{first} {second}"
        for (first, second), _ in sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1])
    ]
    payload = {
        "version": 1,
        "prompt": prompt,
        "context_length": tokenizer.context_length,
        "width": width,
        "heads": width // 64,
        "layers": layers,
        "group_size": GROUP_SIZE,
        "sot": tokenizer.sot_token_id,
        "eot": tokenizer.eot_token_id,
        "vocab": vocab,
        "merges": merges,
        "tensors": tensors,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(IDENTITY_PREFIX + json.dumps(identity, sort_keys=True) + "\n")
        stream.write("window.MEMORY_LANE_TEXT_MODEL = ")
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        stream.write(";\n")
    temporary.replace(path)


def write_text_model(archive: Path, model, tokenizer, identity: dict) -> bool:
    """Export the text model unless an identical one is already present."""
    from .clip_model import TEXT_PROMPT_TEMPLATE

    path = Path(archive) / "assets" / TEXT_MODEL_FILE
    if read_text_model_identity(path) == identity:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().float().cpu().numpy()
        for name, value in model.state_dict().items()
        if not name.startswith("visual.")
    }
    export_text_model(path, state, tokenizer, identity, TEXT_PROMPT_TEMPLATE)
    return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--archive", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    from .browser_index import write_browser_index
    from .clip_model import load_clip, model_identity

    args = parser().parse_args(argv)
    archive = args.archive.resolve()
    try:
        model, _, tokenizer, _ = load_clip(archive / ".model-cache")
        written = write_text_model(archive, model, tokenizer, model_identity())
        config = write_browser_index(archive)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print("Wrote browser text model." if written else "Browser text model is already current.")
    if config["text"] is None:
        print("Warning: the semantic index does not match this model; typed search stays off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
