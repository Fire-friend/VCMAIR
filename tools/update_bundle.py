from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace the three trainable main-model states in a VCMAIR bundle."
    )
    parser.add_argument(
        "--base-bundle", type=Path, default=Path("checkpoints/vcmair.pth")
    )
    parser.add_argument(
        "--main-checkpoint", type=Path, required=True,
        help="Directory containing unet.pth, img_text_mapper.pth, and sd_unet_lora.pth",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/vcmair_trained.pth")
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Check compatibility without writing the multi-gigabyte output",
    )
    return parser.parse_args()


def load_state(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing training weight: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    normalized = {}
    prefixes = ("module.", "online_model.module.", "online_model.")
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        normalized[key] = value
    return normalized


def replace_exact(bundle_state, trained_state, name):
    missing = sorted(set(bundle_state) - set(trained_state))
    unexpected = sorted(set(trained_state) - set(bundle_state))
    if missing or unexpected:
        raise RuntimeError(
            f"{name} keys do not match bundle: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    for key, value in trained_state.items():
        if value.shape != bundle_state[key].shape:
            raise RuntimeError(
                f"{name}.{key} shape mismatch: {value.shape} != {bundle_state[key].shape}"
            )
        bundle_state[key] = value.to(dtype=bundle_state[key].dtype)


def replace_partial(bundle_state, trained_state, name):
    for key, value in trained_state.items():
        if key not in bundle_state:
            raise RuntimeError(f"{name} has an unknown key: {key}")
        if value.shape != bundle_state[key].shape:
            raise RuntimeError(
                f"{name}.{key} shape mismatch: {value.shape} != {bundle_state[key].shape}"
            )
        bundle_state[key] = value.to(dtype=bundle_state[key].dtype)


def main():
    args = parse_args()
    if not args.base_bundle.is_file():
        raise FileNotFoundError(f"Missing base bundle: {args.base_bundle}")
    bundle = torch.load(
        args.base_bundle, map_location="cpu", weights_only=True, mmap=True
    )
    if bundle.get("format") != "vcmair-inference-bundle":
        raise RuntimeError("The base file is not a VCMAIR inference bundle")

    states = bundle["state_dicts"]
    replace_exact(
        states["residual"], load_state(args.main_checkpoint / "unet.pth"), "residual"
    )
    replace_exact(
        states["mapper"],
        load_state(args.main_checkpoint / "img_text_mapper.pth"),
        "mapper",
    )
    replace_partial(
        states["unet"],
        load_state(args.main_checkpoint / "sd_unet_lora.pth"),
        "diffusion LoRA",
    )
    bundle["metadata"]["updated_from"] = str(args.main_checkpoint)
    print("All three training states are compatible with the inference bundle.")
    if args.validate_only:
        return
    if args.output.resolve() == args.base_bundle.resolve():
        raise ValueError("Refusing to overwrite the base bundle; choose another --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Saved self-contained bundle: {args.output}")


if __name__ == "__main__":
    main()
