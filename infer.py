from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from vcmair import VCMAIR


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="All-in-one image restoration with VCMAIR")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output image or directory")
    parser.add_argument(
        "--model", type=Path, default=Path("checkpoints/vcmair.pth"),
        help="Self-contained VCMAIR checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--steps", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_inputs(path: Path):
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input extension: {path.suffix}")
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"Input does not exist: {path}")


def output_path(input_path: Path, input_root: Path, output: Path):
    if input_root.is_file():
        return output if output.suffix else output / f"{input_path.stem}.png"
    relative = input_path.relative_to(input_root).with_suffix(".png")
    return output / relative


def main():
    args = parse_args()
    inputs = collect_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No supported images found in {args.input}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = VCMAIR(
        checkpoint=args.model,
        device=args.device,
        dtype=args.dtype,
        tile_size=args.tile_size,
        overlap=args.overlap,
        steps=args.steps,
        min_size=args.min_size,
    )

    for source in tqdm(inputs, desc="Restoring"):
        destination = output_path(source, args.input, args.output)
        if destination.exists() and not args.overwrite:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            result = model.restore(image, seed=args.seed)
        result.save(destination)


if __name__ == "__main__":
    main()
