import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import CLIPPretrainedModel, CLIPVisionModel

from ...utils import logging


logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = F.normalize(image_embeds)
    normalized_text_embeds = F.normalize(text_embeds)
    return paddle.matmul(normalized_image_embeds, normalized_text_embeds, transpose_y=True)


class StableDiffusionSafetyChecker(CLIPPretrainedModel):
    base_model_class = CLIPVisionModel
    def __init__(self, clip):
        super().__init__()
        self.clip = clip
        projection_dim = clip.config["projection_dim"]
        vision_embed_dim = clip.config["vision_embed_dim"]
        self.vision_projection = paddle.create_parameter(
                (vision_embed_dim, projection_dim), paddle.get_default_dtype())

        self.register_buffer("concept_embeds", paddle.ones([17, projection_dim]))
        self.register_buffer("special_care_embeds", paddle.ones([3, projection_dim]))

        self.register_buffer("concept_embeds_weights", paddle.ones([17]))
        self.register_buffer("special_care_embeds_weights", paddle.ones([3]))

    @paddle.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.clip.vision_model(clip_input)[1]  # pooled_output
        image_embeds = paddle.matmul(pooled_output, self.vision_projection)
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concet_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concet_idx]
                concept_threshold = self.special_care_embeds_weights[concet_idx].item()
                result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concet_idx] > 0:
                    result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
                    adjustment = 0.01

            for concet_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concet_idx]
                concept_threshold = self.concept_embeds_weights[concet_idx].item()
                result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concet_idx] > 0:
                    result_img["bad_concepts"].append(concet_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                images[idx] = np.zeros(images[idx].shape)  # black image

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        return images, has_nsfw_concepts