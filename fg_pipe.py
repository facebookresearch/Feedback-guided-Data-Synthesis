# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F
from diffusers import DiffusionPipeline


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class FeedbackGuidedDiffusion(DiffusionPipeline):
    def __init__(self, vae, text_encoder, clip_model, tokenizer, unet, scheduler, feature_extractor):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            clip_model=clip_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.clip_model, False)

    def process_image(self, sample):
        sample = 1 / self.vae.config.scaling_factor * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5)
        image = image - image.min().detach()
        image = image / image.max().detach()
        return image


    def compute_classifier_fg_criterion(self, fg_criterion, fg_classifier, fg_preprocessing, image, cls_index):
        image_transformed = fg_preprocessing(image)
        logits = fg_classifier(image_transformed)
        if fg_criterion == 'loss':
            y = torch.zeros(*logits.shape)
            y[0, cls_index] = 1
            fg_criterion_func = F.cross_entropy(logits, y.cuda(), reduction='none')
        elif fg_criterion == 'entropy':
            prob = F.softmax(logits, dim=-1)
            fg_criterion_func = F.cross_entropy(logits, prob, reduction='none').mean()

        return fg_criterion_func

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        txt_embd,
        noise_pred_original,
        fg_criterion,
        fg_scale,
        prompt,
        fg_classifier,
        fg_preprocessing,
        cls_index
    ):
        latents = latents.detach().requires_grad_()
        noise_pred = self.unet(latents, timestep, encoder_hidden_states=txt_embd, return_dict=False)[0]

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        fac = torch.sqrt(beta_prod_t)
        sample = pred_original_sample * (fac) + latents * (1 - fac)

        image = self.process_image(sample)
        fg_criterion_func = self.compute_classifier_fg_criterion(fg_criterion, fg_classifier, fg_preprocessing, image, cls_index)

        grads = torch.autograd.grad(fg_criterion_func, latents)[0]
        grads_same_scale = grads / (grads.norm(2).detach() + 1e-8) * latents.norm(2).detach()
        grads = grads_same_scale * fg_scale

        return grads

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        cfg=7.5,
        num_images_per_prompt=1,
        eta=0.0,
        fg_criterion='loss',
        fg_scale=0,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,
        fg_classifier=None,
        guidance_freq=1,
        fg_preprocessing=None,
        cls_index=None
    ):
        batch_size = 1
        if not isinstance(prompt, str):
            raise ValueError(f"input prompt needs to be a string!")

        prompt_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        txt_embd = self.text_encoder(prompt_input.input_ids.to(self.device))[0]
        txt_embd = txt_embd.repeat_interleave(1, dim=0)

        cfg_guidance_ = cfg > 1.0
        if cfg_guidance_:
            max_length = prompt_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            embd_uncond = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            embd_uncond = embd_uncond.repeat_interleave(1, dim=0)
            txt_embd = torch.cat([embd_uncond, txt_embd])

        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = txt_embd.dtype

        self.scheduler.set_timesteps(num_inference_steps, **{})

        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        timestep_index = 0

        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        for i, t in enumerate(self.progress_bar(timesteps_tensor[timestep_index:])):
            latent_model_input = torch.cat([latents] * 2) if cfg_guidance_ else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=txt_embd).sample

            if cfg_guidance_:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

            if fg_scale != 0 and fg_classifier:
                txt_embd_for_guidance = (
                    txt_embd.chunk(2)[1] if cfg_guidance_ else txt_embd
                )
                if i % guidance_freq == 0:
                    grads = self.cond_fn(
                        latents,
                        t,
                        i,
                        txt_embd_for_guidance,
                        noise_pred,
                        fg_criterion,
                        fg_scale,
                        prompt,
                        fg_classifier,
                        fg_preprocessing,
                        cls_index,
                    )
                    latents += grads
            latents = self.scheduler.step(noise_pred, t, latents, **{}, return_dict=False)[0]

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        if fg_preprocessing and fg_classifier:
            criteria_final_value = self.compute_classifier_fg_criterion(fg_criterion, fg_classifier, fg_preprocessing, image, cls_index).mean().item()
            print(fg_criterion, criteria_final_value)
        else:
            criteria_final_value = -10000
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)
        return (image, criteria_final_value)
