from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, TCDScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
    CLIPTextModel,
    CLIPVisionModel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import PairedRestorationDataset
from vcmair.models import LightweightImageToTextTransformerModel, UnetRes
from vcmair.pipeline import ddcm_sampler


logger = get_logger(__name__)
TASKS = ("fog", "light_only", "rain", "snow", "blur")
SOURCE_PROMPT = (
    "fog, blur, blurry, snow, hazy, rain, low quality, low light, dark, "
    "unnatural, unrealistic, cartoon"
)
TARGET_PROMPT = "clear, realistic, high resolution, natural, best quality, fine details, 8K"
LORA_TARGETS = (
    "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
    "conv_shortcut", "time_emb_proj", "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
)


def parse_int_list(value: str, name: str):
    values = [int(item.strip()) for item in value.split(",")]
    if len(values) != len(TASKS) or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            f"{name} must contain {len(TASKS)} positive comma-separated integers"
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the VCMAIR LoRA, residual U-Net, and prompt mapper.\n\n"
            "Fusion-network training is intentionally not part of this release.\n"
        )
    )
    model = parser.add_argument_group("models")
    model.add_argument(
        "--base-model", default="SimianLuo/LCM_Dreamshaper_v7",
        help="Diffusers base model or local directory",
    )
    model.add_argument(
        "--clip-model", default="openai/clip-vit-base-patch32",
        help="CLIP vision model or local directory",
    )
    model.add_argument("--revision", default=None)
    model.add_argument("--lora-rank", type=int, default=32)
    model.add_argument("--lora-alpha", type=int, default=32)
    model.add_argument("--lora-dropout", type=float, default=0.0)

    data = parser.add_argument_group("data")
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument("--light-input-dir", type=Path, default=None)
    data.add_argument("--light-target-dir", type=Path, default=None)
    data.add_argument("--crop-size", type=int, default=256)
    data.add_argument("--load-size", type=int, default=268)
    data.add_argument(
        "--preprocess", choices=("crop", "resize_and_crop"), default="crop"
    )
    data.add_argument("--no-flip", action="store_true")
    data.add_argument("--max-samples-per-task", type=int, default=None)
    data.add_argument("--batch-sizes", default="8,1,4,4,2")
    data.add_argument("--num-workers", default="4,2,2,2,2")

    train = parser.add_argument_group("training")
    train.add_argument("--output-dir", type=Path, default=Path("training_outputs"))
    train.add_argument("--max-train-steps", type=int, default=29000)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--lr-scheduler", default="constant")
    train.add_argument("--lr-warmup-steps", type=int, default=0)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--adam-weight-decay", type=float, default=1e-2)
    train.add_argument("--adam-epsilon", type=float, default=1e-8)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--vae-encode-batch-size", type=int, default=8)
    train.add_argument("--num-ddim-timesteps", type=int, default=4)
    train.add_argument("--preference-margin", type=float, default=0.01)
    train.add_argument("--preference-weight", type=float, default=0.5)
    train.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    train.add_argument("--seed", type=int, default=100)
    train.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--use-8bit-adam", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--xformers", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )

    save = parser.add_argument_group("checkpointing and logging")
    save.add_argument("--checkpointing-steps", type=int, default=500)
    save.add_argument("--checkpoints-total-limit", type=int, default=5)
    save.add_argument("--resume-from-checkpoint", default=None)
    save.add_argument("--report-to", choices=("none", "tensorboard", "wandb"), default="none")
    save.add_argument("--tracker-project-name", default="vcmair-training")
    save.add_argument("--logging-dir", default="logs")

    args = parser.parse_args()
    args.batch_sizes = parse_int_list(args.batch_sizes, "--batch-sizes")
    args.num_workers = parse_int_list(args.num_workers, "--num-workers")
    if args.num_ddim_timesteps < 2:
        parser.error("--num-ddim-timesteps must be at least 2 for preference training")
    return args


def cycle(loader):
    while True:
        yield from loader


def load_state(model, path: Path, strict=True):
    state = torch.load(path, map_location="cpu", weights_only=True)
    prefixes = ("module.", "online_model.module.", "online_model.")
    normalized = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        normalized[key] = value
    incompatible = model.load_state_dict(normalized, strict=strict)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in {path}: {incompatible.unexpected_keys[:5]}")


def save_trainable_models(accelerator, residual, mapper, diffusion_unet, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    residual = accelerator.unwrap_model(residual)
    mapper = accelerator.unwrap_model(mapper)
    diffusion_unet = accelerator.unwrap_model(diffusion_unet)
    torch.save(residual.state_dict(), directory / "unet.pth")
    torch.save(mapper.state_dict(), directory / "img_text_mapper.pth")
    lora = {
        name: parameter.detach().cpu()
        for name, parameter in diffusion_unet.named_parameters()
        if parameter.requires_grad
    }
    torch.save(lora, directory / "sd_unet_lora.pth")


@torch.no_grad()
def encode_fixed_prompt(text_encoder, tokenizer, prompt: str):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    mask = (
        tokens.attention_mask.to(text_encoder.device)
        if getattr(text_encoder.config, "use_attention_mask", False)
        else None
    )
    return text_encoder(
        tokens.input_ids.to(text_encoder.device), attention_mask=mask, return_dict=False
    )[0]


def extract(values, timesteps, shape):
    output = values.gather(-1, timesteps)
    return output.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def find_resume_checkpoint(args):
    if not args.resume_from_checkpoint:
        return None
    if args.resume_from_checkpoint != "latest":
        path = Path(args.resume_from_checkpoint)
        if not path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    candidates = sorted(
        args.output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {args.output_dir}")
    return candidates[-1]


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.output_dir / args.logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=project_config,
    )
    set_seed(args.seed)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(accelerator.state, main_process_only=False)

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    scheduler = TCDScheduler.from_pretrained(
        args.base_model, subfolder="scheduler", revision=args.revision
    )
    scheduler.set_timesteps(args.num_ddim_timesteps, device=accelerator.device)
    alpha_schedule = scheduler.alphas_cumprod.sqrt().to(accelerator.device)
    sigma_schedule = (1 - scheduler.alphas_cumprod).sqrt().to(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model, subfolder="text_encoder", revision=args.revision
    ).requires_grad_(False).eval().to(accelerator.device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae", revision=args.revision
    ).requires_grad_(False).eval().to(accelerator.device)
    clip = CLIPVisionModel.from_pretrained(
        args.clip_model, torch_dtype=dtype
    ).requires_grad_(False).eval().to(accelerator.device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    diffusion_unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", revision=args.revision
    )
    diffusion_unet.requires_grad_(False)
    diffusion_unet.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=list(LORA_TARGETS),
        ),
        adapter_name="default",
    )
    for name, parameter in diffusion_unet.named_parameters():
        parameter.requires_grad_("lora" in name)

    residual = UnetRes(
        dim=64, dim_mults=(1, 2, 4, 8), num_unet=1, condition=True,
        objective="pred_res", test_res_or_noise="res",
    )
    mapper = LightweightImageToTextTransformerModel(
        embedding_dim=768, image_feature_dim=768, num_heads=4, ffn_dim=512,
        num_layers=2, vocab_size=768, dropout=0.0,
    )
    lpips_loss = lpips.LPIPS(net="vgg").requires_grad_(False).eval().to(accelerator.device)
    diffusion_unet.to(accelerator.device)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.xformers:
        if is_xformers_available():
            diffusion_unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xformers is unavailable; continuing without it")
    if args.gradient_checkpointing:
        diffusion_unet.enable_gradient_checkpointing()

    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes is unavailable; using torch.optim.AdamW")
    parameters = (
        list(residual.parameters())
        + list(mapper.parameters())
        + [parameter for parameter in diffusion_unet.parameters() if parameter.requires_grad]
    )
    optimizer = optimizer_class(
        parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    loaders = []
    lengths = []
    for index, task in enumerate(TASKS):
        dataset = PairedRestorationDataset(
            root=args.data_root,
            task=task,
            clip_processor=clip_processor,
            crop_size=args.crop_size,
            load_size=args.load_size,
            preprocess=args.preprocess,
            random_flip=not args.no_flip,
            max_samples=args.max_samples_per_task,
            light_input_dir=args.light_input_dir,
            light_target_dir=args.light_target_dir,
        )
        lengths.append(len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_sizes[index],
            shuffle=True,
            num_workers=args.num_workers[index],
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers[index] > 0,
        )
        if len(loader) == 0:
            raise RuntimeError(
                f"{task}: dataset is smaller than its batch size {args.batch_sizes[index]}"
            )
        loaders.append(loader)
        logger.info("%s: %d training pairs", task, len(dataset))

    residual, mapper, diffusion_unet, optimizer, lr_scheduler, *loaders = accelerator.prepare(
        residual, mapper, diffusion_unet, optimizer, lr_scheduler, *loaders
    )
    loaders = [cycle(loader) for loader in loaders]

    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            save_trainable_models(
                accelerator, residual, mapper, diffusion_unet, Path(output_dir)
            )
        weights.clear()

    def load_hook(models, input_dir):
        load_state(accelerator.unwrap_model(residual), Path(input_dir) / "unet.pth")
        load_state(
            accelerator.unwrap_model(mapper), Path(input_dir) / "img_text_mapper.pth"
        )
        load_state(
            accelerator.unwrap_model(diffusion_unet),
            Path(input_dir) / "sd_unet_lora.pth",
            strict=False,
        )
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    if accelerator.is_main_process and args.report_to != "none":
        tracker_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    source_embed = encode_fixed_prompt(text_encoder, tokenizer, SOURCE_PROMPT).to(dtype=dtype)
    target_embed = encode_fixed_prompt(text_encoder, tokenizer, TARGET_PROMPT).to(dtype=dtype)
    del tokenizer, text_encoder, clip_processor

    global_step = 0
    resume_path = find_resume_checkpoint(args)
    if resume_path:
        accelerator.load_state(resume_path)
        global_step = int(resume_path.name.rsplit("-", 1)[1])
        logger.info("Resumed from %s at step %d", resume_path, global_step)

    residual.train()
    mapper.train()
    diffusion_unet.train()
    progress = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        total=args.max_train_steps,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    autocast = (
        torch.autocast(accelerator.device.type, dtype=dtype)
        if accelerator.mixed_precision != "no"
        else nullcontext()
    )

    while global_step < args.max_train_steps:
        batches = [next(loader) for loader in loaders]
        lq = torch.cat([batch["lq"] for batch in batches]).to(
            accelerator.device, non_blocking=True
        )
        gt = torch.cat([batch["gt"] for batch in batches]).to(
            accelerator.device, non_blocking=True
        )
        clip_input = torch.cat([batch["clip_input"] for batch in batches]).to(
            accelerator.device, non_blocking=True
        )

        with accelerator.accumulate(residual, mapper, diffusion_unet):
            pixel_values = lq.to(dtype=dtype) * 2.0 - 1.0
            with torch.no_grad():
                latent_parts = []
                for start in range(0, len(pixel_values), args.vae_encode_batch_size):
                    distribution = vae.encode(
                        pixel_values[start : start + args.vae_encode_batch_size].float()
                    ).latent_dist
                    latent_parts.append(distribution.sample())
                latents = torch.cat(latent_parts).mul_(vae.config.scaling_factor).to(dtype)
                with autocast:
                    clip_features = clip(clip_input.to(dtype=dtype)).last_hidden_state[:, 0]
            batch_size = latents.shape[0]
            source_prompts = source_embed.expand(batch_size, -1, -1)
            target_prompts = target_embed.expand(batch_size, -1, -1)

            scheduler.set_timesteps(args.num_ddim_timesteps, device=accelerator.device)
            schedule = scheduler.timesteps.long()
            predictions = []
            source_noisy = None
            target_noisy = None
            for step_index in range(2):
                timesteps = schedule[step_index].expand(batch_size)
                alpha = extract(alpha_schedule, timesteps, latents.shape)
                sigma = extract(sigma_schedule, timesteps, latents.shape)
                noise = torch.randn_like(latents)
                if step_index == 0:
                    source_noisy = scheduler.add_noise(latents, noise, timesteps)
                    target_noisy = source_noisy

                with autocast:
                    source_condition = mapper(source_prompts, clip_features)
                    target_condition = mapper(target_prompts, clip_features)
                    source_residual = residual(
                        torch.cat([source_noisy, latents], dim=1).to(dtype), timesteps
                    )
                    target_residual = residual(
                        torch.cat([target_noisy, latents], dim=1).to(dtype), timesteps
                    )
                    corrected_source = alpha * latents + sigma * source_residual
                    corrected_target = alpha * latents + sigma * target_residual
                    source_noise = diffusion_unet(
                        corrected_source,
                        timesteps,
                        encoder_hidden_states=source_condition,
                        return_dict=False,
                    )[0]
                    target_noise = diffusion_unet(
                        corrected_target,
                        timesteps,
                        encoder_hidden_states=target_condition,
                        return_dict=False,
                    )[0]
                    _, _, prediction = ddcm_sampler(
                        scheduler=scheduler,
                        x_s=corrected_source,
                        x_t=corrected_target,
                        timestep=timesteps[0],
                        e_s=source_noise,
                        e_t=target_noise,
                        x_0=latents,
                        noise=noise,
                        eta=1.0,
                        to_next=True,
                    )
                predictions.append(prediction)
                latents = prediction

            image_one = vae.decode(
                predictions[0].float() / vae.config.scaling_factor, return_dict=False
            )[0].mul(0.5).add(0.5)
            image_two = vae.decode(
                predictions[1].float() / vae.config.scaling_factor, return_dict=False
            )[0].mul(0.5).add(0.5)

            one_l1 = F.l1_loss(image_one.float(), gt.float())
            one_lpips = lpips_loss(image_one.float(), gt.float()).mean()
            one_score = one_l1 + one_lpips
            two_score = F.l1_loss(image_two.float(), gt.float()) + lpips_loss(
                image_two.float(), gt.float()
            ).mean()
            preference = F.relu(
                two_score - one_score.detach() + args.preference_margin
            )
            loss = one_score + args.preference_weight * preference

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            progress.update(1)
            logs = {
                "loss": loss.detach().item(),
                "loss_l1": one_l1.detach().item(),
                "loss_lpips": one_lpips.detach().item(),
                "loss_preference": preference.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process and args.checkpoints_total_limit > 0:
                    checkpoints = sorted(
                        args.output_dir.glob("checkpoint-*"),
                        key=lambda path: int(path.name.rsplit("-", 1)[1]),
                    )
                    while len(checkpoints) >= args.checkpoints_total_limit:
                        shutil.rmtree(checkpoints.pop(0))
                accelerator.wait_for_everyone()
                save_path = args.output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(save_path)
                if accelerator.is_main_process:
                    samples = save_path / "samples"
                    samples.mkdir(exist_ok=True)
                    save_image(image_one.detach(), samples / "pred.png")
                    save_image(gt.detach(), samples / "gt.png")
                    save_image(lq.detach(), samples / "lq.png")
                logger.info("Saved checkpoint to %s", save_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_trainable_models(
            accelerator, residual, mapper, diffusion_unet, args.output_dir / "final_model"
        )
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
