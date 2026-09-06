from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, TCDScheduler, UNet2DConditionModel
from peft import LoraConfig
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from .models import (
    LightweightImageToTextTransformerModel,
    NAFNet_Combine,
    UnetRes,
)
from .pipeline import StableDiffusionRestorePipeline


def _load_exact(model: torch.nn.Module, state: dict, name: str):
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Invalid {name} state dict")
    return model


def _gaussian_weights(tile_latent_size: int, batch: int, device, dtype):
    variance = 0.01
    midpoint_x = (tile_latent_size - 1) / 2
    midpoint_y = tile_latent_size / 2
    xs = np.asarray(
        [
            np.exp(-((x - midpoint_x) ** 2) / (tile_latent_size**2) / (2 * variance))
            / np.sqrt(2 * np.pi * variance)
            for x in range(tile_latent_size)
        ]
    )
    ys = np.asarray(
        [
            np.exp(-((y - midpoint_y) ** 2) / (tile_latent_size**2) / (2 * variance))
            / np.sqrt(2 * np.pi * variance)
            for y in range(tile_latent_size)
        ]
    )
    weight = torch.as_tensor(np.outer(ys, xs), device=device, dtype=dtype)
    return weight.expand(batch, 4, -1, -1)


def _pad_image(tensor, min_size: int):
    _, _, height, width = tensor.shape
    target_h = max(min_size, height)
    target_w = max(min_size, width)
    target_h = (target_h + 7) // 8 * 8
    target_w = (target_w + 7) // 8 * 8
    pad_h, pad_w = target_h - height, target_w - width
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    mode = "reflect" if max(padding[0], padding[1]) < width and max(padding[2], padding[3]) < height else "replicate"
    return F.pad(tensor, padding, mode=mode), (height, width)


def _crop_image(tensor, original_size):
    height, width = original_size
    start_h = (tensor.shape[-2] - height) // 2
    start_w = (tensor.shape[-1] - width) // 2
    return tensor[..., start_h : start_h + height, start_w : start_w + width]


class VCMAIR:
    def __init__(
        self,
        checkpoint: str | Path = "checkpoints/vcmair.pth",
        device: str = "cuda",
        dtype: str = "float16",
        tile_size: int = 256,
        overlap: int = 16,
        steps: int = 1,
        min_size: int = 256,
    ):
        if tile_size % 8 or overlap % 8 or tile_size <= overlap:
            raise ValueError("tile_size and overlap must be multiples of 8, and tile_size > overlap")
        if not 1 <= steps <= 4:
            raise ValueError("steps must be in [1, 4]")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.dtype = dtype_map[dtype]
        if self.device.type == "cpu" and self.dtype == torch.float16:
            raise ValueError("Use --dtype float32 for CPU debugging")
        self.tile_size = tile_size
        self.overlap = overlap
        self.steps = steps
        self.min_size = max(min_size, tile_size)

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing VCMAIR checkpoint: {checkpoint}")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if bundle.get("format") != "vcmair-inference-bundle" or bundle.get("format_version") != 1:
            raise RuntimeError(f"Unsupported VCMAIR checkpoint format: {checkpoint}")
        configs = bundle["configs"]
        states = bundle["state_dicts"]

        vae = AutoencoderKL.from_config(configs["vae"]).to(dtype=self.dtype)
        _load_exact(vae, states["vae"], "VAE")
        unet = UNet2DConditionModel.from_config(configs["unet"])
        unet.add_adapter(
            LoraConfig(
                r=32,
                lora_alpha=32,
                init_lora_weights="gaussian",
                target_modules=[
                    "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
                    "conv_shortcut", "time_emb_proj", "proj_in", "proj_out",
                    "ff.net.0.proj", "ff.net.2",
                ],
            )
        )
        unet.to(dtype=self.dtype)
        _load_exact(unet, states["unet"], "diffusion U-Net")
        unet.eval()

        scheduler = TCDScheduler.from_config(configs["scheduler"])
        self.pipeline = StableDiffusionRestorePipeline(
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False,
        ).to(self.device)

        residual = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            num_unet=1,
            condition=True,
            objective="pred_res",
            test_res_or_noise="res",
        )
        mapper = LightweightImageToTextTransformerModel(
            embedding_dim=768,
            image_feature_dim=768,
            num_heads=4,
            ffn_dim=512,
            num_layers=2,
            vocab_size=768,
            dropout=0.1,
        )
        fusion = NAFNet_Combine(
            img_channel=6,
            width=64,
            enc_blk_nums=(2, 2, 4, 8),
            middle_blk_num=12,
            dec_blk_nums=(2, 2, 2, 2),
        )
        self.pipeline.residual_model = _load_exact(
            residual, states["residual"], "residual U-Net"
        ).to(self.device).eval()
        self.pipeline.prompt_mapper = _load_exact(
            mapper, states["mapper"], "prompt mapper"
        ).to(self.device).eval()
        self.fusion = _load_exact(fusion, states["fusion"], "fusion network").to(self.device).eval()

        self.source_prompt_embeds = bundle["prompt_embeds"]["source"].to(
            self.device, dtype=self.dtype
        )
        self.target_prompt_embeds = bundle["prompt_embeds"]["target"].to(
            self.device, dtype=self.dtype
        )
        self.clip_processor = CLIPImageProcessor.from_dict(configs["clip_image_processor"])
        clip_config = CLIPVisionConfig.from_dict(configs["clip"])
        self.clip = CLIPVisionModel(clip_config).to(dtype=self.dtype)
        _load_exact(self.clip, states["clip"], "CLIP vision encoder")
        self.clip.to(self.device).eval()
        del bundle

    @torch.inference_mode()
    def restore(self, image: Image.Image, seed: int = 0) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor, original_size = _pad_image(tensor, self.min_size)
        restored = self._restore_tiled(tensor, seed=seed)
        restored = self.fusion(torch.cat([tensor, restored], dim=1)).clamp(0, 1)
        restored = _crop_image(restored, original_size)[0].float().cpu()
        output = (restored.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(output, mode="RGB")

    def _restore_tiled(self, image, seed: int):
        batch, _, height, width = image.shape
        stride = self.tile_size - self.overlap
        latent_h, latent_w = height // 8, width // 8
        output = torch.zeros((batch, 4, latent_h, latent_w), device=self.device, dtype=torch.float32)
        weight_map = torch.zeros_like(output)
        weights = _gaussian_weights(
            self.tile_size // 8, batch, self.device, torch.float32
        )

        row_starts = list(range(0, max(height - self.tile_size, 0) + 1, stride))
        col_starts = list(range(0, max(width - self.tile_size, 0) + 1, stride))
        if row_starts[-1] != height - self.tile_size:
            row_starts.append(height - self.tile_size)
        if col_starts[-1] != width - self.tile_size:
            col_starts.append(width - self.tile_size)

        for top in row_starts:
            for left in col_starts:
                tile = image[..., top : top + self.tile_size, left : left + self.tile_size]
                clip_inputs = self.clip_processor(
                    images=(tile * 255.0).long(), return_tensors="pt"
                ).to(self.device)
                clip_features = self.clip(**clip_inputs).last_hidden_state[:, 0]
                generator = torch.Generator(device="cpu").manual_seed(seed)
                (latent,) = self.pipeline(
                    image=tile,
                    clip_features=clip_features,
                    source_prompt_embeds=self.source_prompt_embeds,
                    target_prompt_embeds=self.target_prompt_embeds,
                    num_inference_steps=self.steps,
                    generator=generator,
                    output_type="latent",
                )
                y0, y1 = top // 8, (top + self.tile_size) // 8
                x0, x1 = left // 8, (left + self.tile_size) // 8
                output[..., y0:y1, x0:x1] += latent.float() * weights
                weight_map[..., y0:y1, x0:x1] += weights

        latent = output / weight_map.clamp_min(1e-8)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            decoded = self.pipeline.vae.decode(
                latent / self.pipeline.vae.config.scaling_factor, return_dict=False
            )[0]
        return (decoded * 0.5 + 0.5)[..., :height, :width]
