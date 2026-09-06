from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


TASK_DIRECTORIES = {
    "fog": ("RESIDE/OTS_ALPHA/haze/OTS", "RESIDE/OTS_ALPHA/clear/clear_images"),
    "light_only": ("RNI15", "RNI15"),
    "rain": ("syn_rain/train/input", "syn_rain/train/target"),
    "snow": ("Snow100K/train/synthetic", "Snow100K/train/gt"),
    "blur": ("Deblur/train/input", "Deblur/train/target"),
}


def _images(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _pair_images(task: str, input_dir: Path, target_dir: Path):
    inputs = _images(input_dir)
    targets = _images(target_dir)
    if task == "light_only" and input_dir.resolve() == target_dir.resolve():
        return [(path, path) for path in inputs]

    target_map = {path.stem: path for path in targets}
    pairs = []
    missing = []
    for source in inputs:
        target_stem = source.stem.split("_")[0] if task == "fog" else source.stem
        target = target_map.get(target_stem)
        if target is None:
            missing.append(source.name)
        else:
            pairs.append((source, target))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{task}: {len(missing)} inputs have no target: {preview}")
    if not pairs:
        raise RuntimeError(f"{task}: no paired images found")
    return pairs


class PairedRestorationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        task: str,
        clip_processor,
        crop_size: int = 256,
        load_size: int = 268,
        preprocess: str = "crop",
        random_flip: bool = True,
        max_samples: int | None = None,
        light_input_dir: str | Path | None = None,
        light_target_dir: str | Path | None = None,
    ):
        if task not in TASK_DIRECTORIES:
            raise ValueError(f"Unknown restoration task: {task}")
        root = Path(root)
        input_rel, target_rel = TASK_DIRECTORIES[task]
        if task == "light_only" and light_input_dir is not None:
            input_dir = Path(light_input_dir)
            target_dir = Path(light_target_dir or light_input_dir)
        else:
            input_dir, target_dir = root / input_rel, root / target_rel
        self.task = task
        self.pairs = _pair_images(task, input_dir, target_dir)
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        self.clip_processor = clip_processor
        self.crop_size = crop_size
        self.load_size = load_size
        self.preprocess = preprocess
        self.random_flip = random_flip

    def __len__(self):
        return len(self.pairs)

    def _paired_transform(self, source: Image.Image, target: Image.Image):
        if source.size != target.size:
            raise RuntimeError(
                f"Unaligned pair has different sizes: {source.size} and {target.size}"
            )
        if "resize" in self.preprocess:
            size = [self.load_size, self.load_size]
            source = TF.resize(source, size, InterpolationMode.BICUBIC)
            target = TF.resize(target, size, InterpolationMode.BICUBIC)

        width, height = source.size
        if "crop" in self.preprocess and (width > self.crop_size or height > self.crop_size):
            left = random.randint(0, max(0, width - self.crop_size))
            top = random.randint(0, max(0, height - self.crop_size))
            source = TF.crop(source, top, left, self.crop_size, self.crop_size)
            target = TF.crop(target, top, left, self.crop_size, self.crop_size)
        if self.random_flip and random.random() > 0.5:
            source = TF.hflip(source)
            target = TF.hflip(target)

        source = TF.to_tensor(source)
        target = TF.to_tensor(target)
        if source.shape[-2:] != (self.crop_size, self.crop_size):
            size = [self.crop_size, self.crop_size]
            source = TF.resize(source, size, InterpolationMode.BICUBIC, antialias=True)
            target = TF.resize(target, size, InterpolationMode.BICUBIC, antialias=True)
        return source.clamp(0, 1), target.clamp(0, 1)

    def __getitem__(self, index):
        source_path, target_path = self.pairs[index]
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(target_path) as image:
            target = image.convert("RGB")
        source, target = self._paired_transform(source, target)
        clip_input = self.clip_processor(
            images=(source * 255.0).long(), return_tensors="pt"
        ).pixel_values[0]
        return {
            "lq": source,
            "gt": target,
            "clip_input": clip_input,
            "lq_path": str(source_path),
            "gt_path": str(target_path),
        }
