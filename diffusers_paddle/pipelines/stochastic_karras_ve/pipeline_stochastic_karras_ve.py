#!/usr/bin/env python3
from typing import Optional, Tuple, Union

import paddle

from ...models import UNet2DModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import KarrasVeScheduler


class KarrasVePipeline(DiffusionPipeline):
    r"""
    Stochastic sampling from Karras et al. [1] tailored to the Variance-Expanding (VE) models [2]. Use Algorithm 2 and
    the VE column of Table 1 from [1] for reference.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364 [2] Song, Yang, et al. "Score-based generative modeling through stochastic
    differential equations." https://arxiv.org/abs/2011.13456

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`KarrasVeScheduler`]):
            Scheduler for the diffusion process to be used in combination with `unet` to denoise the encoded image.
    """

    # add type hints for linting
    unet: UNet2DModel
    scheduler: KarrasVeScheduler

    def __init__(self, unet: UNet2DModel, scheduler: KarrasVeScheduler):
        super().__init__()
        scheduler = scheduler.set_format("pd")
        self.register_modules(unet=unet, scheduler=scheduler)

    @paddle.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (:obj:`int`, *optional*, defaults to 1):
                The number of images to generate.
            seed (:obj:`int`, *optional*):
                Only seed.
            num_inference_steps (:obj:`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (:obj:`str`, *optional*, defaults to :obj:`"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (:obj:`bool`, *optional*, defaults to :obj:`True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        """

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet
        # TODO add global seed
        if seed is not None:
            paddle.seed(seed)
        # sample x_0 ~ N(0, sigma_0^2 * I)
        sample = paddle.randn(shape) * self.scheduler.config.sigma_max
        sample = sample

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # here sigma_t == t_i from the paper
            sigma = self.scheduler.schedule[t]
            sigma_prev = self.scheduler.schedule[t - 1] if t > 0 else 0

            # 1. Select temporarily increased noise level sigma_hat
            # 2. Add new noise to move from sample_i to sample_hat
            sample_hat, sigma_hat = self.scheduler.add_noise_to_input(sample, sigma) 

            # 3. Predict the noise residual given the noise magnitude `sigma_hat`
            # The model inputs and output are adjusted by following eq. (213) in [1].
            model_output = (sigma_hat / 2) * model((sample_hat + 1) / 2, sigma_hat / 2).sample

            # 4. Evaluate dx/dt at sigma_hat
            # 5. Take Euler step from sigma to sigma_prev
            step_output = self.scheduler.step(model_output, sigma_hat, sigma_prev, sample_hat)

            if sigma_prev != 0:
                # 6. Apply 2nd order correction
                # The model inputs and output are adjusted by following eq. (213) in [1].
                model_output = (sigma_prev / 2) * model((step_output.prev_sample + 1) / 2, sigma_prev / 2).sample
                step_output = self.scheduler.step_correct(
                    model_output,
                    sigma_hat,
                    sigma_prev,
                    sample_hat,
                    step_output.prev_sample,
                    step_output["derivative"],
                )
            sample = step_output.prev_sample

        sample = (sample / 2 + 0.5).clip(0, 1)
        image = sample.transpose([0, 2, 3, 1]).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(sample)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
