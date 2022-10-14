# flake8: noqa
from ...utils import is_paddlenlp_available


if is_paddlenlp_available():
    from .pipeline_latent_diffusion import LDMBertModel, LDMTextToImagePipeline
