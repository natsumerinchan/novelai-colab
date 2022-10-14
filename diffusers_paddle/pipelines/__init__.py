from ..utils import is_paddlenlp_available
from .ddim import DDIMPipeline
from .ddpm import DDPMPipeline
from .latent_diffusion_uncond import LDMPipeline
from .pndm import PNDMPipeline
from .score_sde_ve import ScoreSdeVePipeline
from .stochastic_karras_ve import KarrasVePipeline


if is_paddlenlp_available():
    from .latent_diffusion import LDMTextToImagePipeline
    from .stable_diffusion import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
        ErnieStableDiffusionPipeline
    )