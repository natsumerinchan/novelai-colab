#!/usr/bin/env python3
from typing import Optional, Tuple, Union

import paddle

from ...models import UNet2DModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import ScoreSdeVeScheduler


class ScoreSdeVePipeline(DiffusionPipeline):
    r"""
    Parameters:
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image. scheduler ([`SchedulerMixin`]):
            The [`ScoreSdeVeScheduler`] scheduler to be used in combination with `unet` to denoise the encoded image.
    """
    unet: UNet2DModel
    scheduler: ScoreSdeVeScheduler

    def __init__(self, unet: UNet2DModel, scheduler: DiffusionPipeline):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @paddle.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        seed: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (:obj:`int`, *optional*, defaults to 1):
                The number of images to generate.
            seed (:obj:`int`, *optional*):
                Only seed.
            output_type (:obj:`str`, *optional*, defaults to :obj:`"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (:obj:`bool`, *optional*, defaults to :obj:`True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        """


        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet
        if seed is not None:
            paddle.seed(seed)
        sample = paddle.randn(shape) * self.scheduler.config.sigma_max
        sample = sample

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            sigma_t = self.scheduler.sigmas[i] * paddle.ones([shape[0]])

            # correction step
            for _ in range(self.scheduler.config.correct_steps):
                model_output = self.unet(sample, sigma_t).sample
                sample = self.scheduler.step_correct(model_output, sample).prev_sample

            # prediction step
            model_output = model(sample, sigma_t).sample
            output = self.scheduler.step_pred(model_output, t, sample)

            sample, sample_mean = output.prev_sample, output.prev_sample_mean

        sample = sample_mean.clip(0, 1)
        sample = sample.transpose([0, 2, 3, 1]).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)
