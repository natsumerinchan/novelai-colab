from . import patch_utils

from .utils import (
    is_inflect_available,
    is_scipy_available,
    is_paddle_available,
    is_paddlenlp_available,
    is_unidecode_available,
)


__version__ = "0.3.0"

from .modeling_utils import ModelMixin
from .models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
from .optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_scheduler,
)
from .pipeline_utils import DiffusionPipeline
from .pipelines import DDIMPipeline, DDPMPipeline, KarrasVePipeline, LDMPipeline, PNDMPipeline, ScoreSdeVePipeline
from .schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    KarrasVeScheduler,
    PNDMScheduler,
    SchedulerMixin,
    ScoreSdeVeScheduler,
)
from .utils import logging


if is_scipy_available():
    from .schedulers import LMSDiscreteScheduler
else:
    from .utils.dummy_scipy_objects import *  # noqa F403

from .training_utils import EMAModel


if is_paddlenlp_available():
    from .pipelines import (
        LDMTextToImagePipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
    )
else:
    from .utils.dummy_paddlenlp_objects import *  # noqa F403