# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# limitations under the License.


from typing import Optional, Tuple, Union

import paddle

from ...models import UNet2DModel
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...schedulers import PNDMScheduler


class PNDMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet (:obj:`UNet2DModel`): U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            The `PNDMScheduler` to be used in combination with `unet` to denoise the encoded image.
    """

    unet: UNet2DModel
    scheduler: PNDMScheduler

    def __init__(self, unet: UNet2DModel, scheduler: PNDMScheduler):
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
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (:obj:`int`, `optional`, defaults to 1): The number of images to generate.
            num_inference_steps (:obj:`int`, `optional`, defaults to 50): The number of denoising steps. More denoising steps usually
                lead to a higher quality image at the expense of slower inference.
            seed (:obj:`int`, `optional`):
                Only seed.
            output_type (:obj:`str`, `optional`, defaults to :obj:`"pil"`): The output format of the generate image. Choose
                between [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `nd.array`.
            return_dict (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether or not to return a
                [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        """
        # For more information on the sampling method you can take a look at Algorithm 2 of
        # the official paper: https://arxiv.org/pdf/2202.09778.pdf

        if seed is not None:
            paddle.seed(seed)
        # Sample gaussian noise to begin loop
        image = paddle.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
        )
        image = image

        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(image, t).sample

            image = self.scheduler.step(model_output, t, image).prev_sample

        image = (image / 2 + 0.5).clip(0, 1)
        image = image.transpose([0, 2, 3, 1]).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
