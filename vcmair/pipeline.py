from __future__ import annotations

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
    retrieve_latents,
    retrieve_timesteps,
)
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor


def ddcm_sampler(
    scheduler,
    x_s,
    x_t,
    timestep,
    e_s,
    e_t,
    x_0,
    noise,
    eta=1.0,
    to_next=True,
    residual_scale=1.0,
):
    if scheduler.num_inference_steps is None:
        raise ValueError("scheduler.set_timesteps(...) must be called before sampling")
    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    next_index = scheduler.step_index + 1
    prev_timestep = (
        scheduler.timesteps[next_index]
        if next_index < len(scheduler.timesteps)
        else timestep
    )
    alpha_t = scheduler.alphas_cumprod[timestep]
    alpha_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    beta_t = 1 - alpha_t
    beta_prev = 1 - alpha_prev
    noise = (eta * beta_prev).sqrt() * noise

    correction = (x_s - alpha_t.sqrt() * x_0) / beta_t.sqrt()
    pred_x0 = x_0 + (
        (x_t - x_s) - beta_t.sqrt() * residual_scale * (e_t - e_s)
    ) / alpha_t.sqrt()
    epsilon = residual_scale * (e_t - e_s) + correction
    direction = (beta_prev - eta * beta_prev).sqrt() * epsilon

    if len(scheduler.timesteps) > 1:
        prev_x_t = alpha_prev.sqrt() * pred_x0 + direction + noise
        prev_x_s = alpha_prev.sqrt() * x_0 + direction + noise
    else:
        prev_x_t, prev_x_s = pred_x0, x_0

    if to_next:
        scheduler._step_index += 1
    return prev_x_s, prev_x_t, pred_x0


class StableDiffusionRestorePipeline(StableDiffusionImg2ImgPipeline):
    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
    ):
        if not isinstance(image, torch.Tensor):
            raise TypeError("VCMAIR expects a preprocessed NCHW torch.Tensor")
        image = image.to(device=device, dtype=dtype)
        batch_size *= num_images_per_prompt

        if image.shape[1] == 4:
            latents = image
        elif isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError("The generator list length must equal the effective batch size")
            latents = torch.cat(
                [retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i]) for i in range(batch_size)]
            )
            latents = self.vae.config.scaling_factor * latents
        else:
            latents = self.vae.config.scaling_factor * retrieve_latents(
                self.vae.encode(image), generator=generator
            )

        if batch_size > latents.shape[0] and batch_size % latents.shape[0] == 0:
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                "Duplicating image latents to match the number of prompts is deprecated.",
                standard_warn=False,
            )
            latents = torch.cat([latents] * (batch_size // latents.shape[0]))
        elif batch_size != latents.shape[0]:
            raise ValueError("Image batch size must match prompt batch size")

        clean_latents = latents.clone()
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)
        return self.scheduler.add_noise(latents, noise, timestep), clean_latents

    @torch.inference_mode()
    def __call__(
        self,
        *,
        image: torch.Tensor,
        clip_features: torch.Tensor,
        source_prompt_embeds: torch.Tensor,
        target_prompt_embeds: torch.Tensor,
        num_inference_steps: int = 1,
        generator=None,
        output_type: str = "latent",
        return_dict: bool = False,
    ):
        if not 1 <= num_inference_steps <= 4:
            raise ValueError("num_inference_steps must be between 1 and 4")
        if not hasattr(self, "residual_model") or not hasattr(self, "prompt_mapper"):
            raise RuntimeError("VCMAIR auxiliary models are not attached to the pipeline")

        self._guidance_scale = 0.0
        self._clip_skip = None
        self._cross_attention_kwargs = None
        self._interrupt = False
        device = self._execution_device
        batch_size = image.shape[0]
        source_embeds = source_prompt_embeds.to(device=device, dtype=self.unet.dtype)
        target_embeds = target_prompt_embeds.to(device=device, dtype=self.unet.dtype)
        if source_embeds.shape[0] == 1 and batch_size > 1:
            source_embeds = source_embeds.repeat(batch_size, 1, 1)
        if target_embeds.shape[0] == 1 and batch_size > 1:
            target_embeds = target_embeds.repeat(batch_size, 1, 1)
        if source_embeds.shape[0] != batch_size or target_embeds.shape[0] != batch_size:
            raise ValueError("Prompt-embedding batch size does not match the image batch")
        image = self.image_processor.preprocess(image)

        timesteps, schedule_steps = retrieve_timesteps(self.scheduler, 4, device)
        timesteps, schedule_steps = self.get_timesteps(schedule_steps, 1.0, device)
        latent_timestep = timesteps[:1].repeat(batch_size)
        timesteps = timesteps[:num_inference_steps]
        self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            1,
            source_embeds.dtype,
            device,
            generator,
        )
        latents, lq_x0 = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            1,
            source_embeds.dtype,
            device,
            generator,
        )

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance = torch.full((batch_size,), -1.0)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        x_source = latents
        x_target = latents
        pred_x0 = lq_x0
        autocast_enabled = device.type == "cuda"
        for timestep in timesteps:
            alpha = self.scheduler.alphas_cumprod[int(timestep.item())]
            source_input = self.scheduler.scale_model_input(x_source, timestep)
            target_input = self.scheduler.scale_model_input(x_target, timestep)

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                source_condition = self.prompt_mapper(
                    source_embeds.detach()[..., :768],
                    clip_features.detach(),
                    t=timestep.repeat(batch_size),
                )
                target_condition = self.prompt_mapper(
                    target_embeds.detach()[..., :768],
                    clip_features.detach(),
                    t=timestep.repeat(batch_size),
                )
                source_noise = self.residual_model(
                    torch.cat([source_input, lq_x0], dim=1), timestep.reshape(-1)
                )
                x_source = alpha.sqrt() * lq_x0 + (1 - alpha).sqrt() * source_noise
                target_noise = self.residual_model(
                    torch.cat([target_input, lq_x0], dim=1), timestep.reshape(-1)
                )
                x_target = alpha.sqrt() * lq_x0 + (1 - alpha).sqrt() * target_noise
                source_input = self.scheduler.scale_model_input(x_source, timestep)
                target_input = self.scheduler.scale_model_input(x_target, timestep)
                source_prediction = self.unet(
                    source_input,
                    timestep,
                    encoder_hidden_states=source_condition,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]
                target_prediction = self.unet(
                    target_input,
                    timestep,
                    encoder_hidden_states=target_condition,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]

            noise = randn_tensor(
                latents.shape,
                dtype=latents.dtype,
                device=latents.device,
                generator=generator,
            )
            x_source, x_target, pred_x0 = ddcm_sampler(
                self.scheduler,
                x_source,
                x_target,
                timestep,
                source_prediction,
                target_prediction,
                lq_x0,
                noise,
            )
            x_source = x_source.detach()
            x_target = x_target.detach()
            lq_x0 = pred_x0.detach()

        if output_type == "latent":
            return (pred_x0,) if not return_dict else {"latents": pred_x0}

        decoded = self.vae.decode(
            pred_x0 / self.vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        decoded = self.image_processor.postprocess(
            decoded,
            output_type=output_type,
            do_denormalize=[True] * decoded.shape[0],
        )
        if not return_dict:
            return (decoded,)
        return StableDiffusionPipelineOutput(images=decoded)
