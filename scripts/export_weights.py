"""Export PyTorch state_dict weights to ``models/ann_weights.npz``.

This script is a maintenance utility.  It does not retrain anything; it simply
copies the tensors from the saved ``.pth`` checkpoint into NumPy arrays that the
PyTorch-free inference module can load.

Usage
-----
    python scripts/export_weights.py
    python scripts/export_weights.py --checkpoint models/ann_customer_churn.pth \
        --output models/ann_weights.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "ann_customer_churn.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "ann_weights.npz"

WEIGHT_KEYS = (
    "network.0.weight",
    "network.0.bias",
    "network.2.weight",
    "network.2.bias",
    "network.4.weight",
    "network.4.bias",
)


def load_state_dict(checkpoint_path: Path) -> dict[str, np.ndarray]:
    """Load a PyTorch state_dict and return float32 NumPy arrays."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "PyTorch is required only for exporting a checkpoint. "
            "Install the training dependencies with "
            "'pip install -r requirements-train.txt'."
        ) from error

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    arrays: dict[str, np.ndarray] = {}
    for key in WEIGHT_KEYS:
        if key not in state_dict:
            raise KeyError(f"{checkpoint_path} does not contain tensor {key!r}")
        arrays[key] = (
            state_dict[key].detach().cpu().numpy().astype(np.float32, copy=True)
        )

    return arrays


def export_weights(checkpoint_path: Path, output_path: Path) -> dict[str, np.ndarray]:
    """Export a checkpoint to ``.npz`` and verify the written file."""

    arrays = load_state_dict(checkpoint_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)

    # Re-open the file and verify that every array round-trips exactly.
    with np.load(output_path, allow_pickle=False) as exported:
        for key, expected in arrays.items():
            if not np.array_equal(exported[key], expected):
                raise RuntimeError(f"Verification failed for {key!r}")

    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the PyTorch .pth checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination .npz file",
    )
    args = parser.parse_args()

    arrays = export_weights(args.checkpoint, args.output)

    print(f"Exported {len(arrays)} tensors to {args.output}")
    for key, value in arrays.items():
        print(f"  {key:20s} shape={value.shape} dtype={value.dtype}")


if __name__ == "__main__":
    main()
