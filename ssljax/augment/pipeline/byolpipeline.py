from collections import OrderedDict
import collections

import jax
import jax.numpy as jnp
from ssljax.augment.pipeline.pipeline import Pipeline
from ssljax.augment.augmentation.augmentation import AugmentationDistribution, Clip, ColorTransform, RandomFlip, RandomGaussianBlur, Solarize
from ssljax.core.utils import register


# TODO: clip is followed by stop_grad
byolaugmentations = collections.OrderedDict({0: [
        RandomFlip(prob=1.0),
        ColorTransform(
            prob=1.0,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            color_jitter_prob=0.8,
            to_grayscale_prob=0.2,
            shuffle=True,
        ),
        RandomGaussianBlur(
            prob=1.0,
            kernel_size=3,
            padding=0,
            sigma_min=0.1,
            sigma_max=2.0,
        ),
        Clip(0,1),
    ],
    1: [
        RandomFlip(prob=1.0),
        ColorTransform(
            prob=1.0,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            color_jitter_prob=0.8,
            to_grayscale_prob=0.2,
            shuffle=True,
        ),
        RandomGaussianBlur(
            prob=0.1,
            kernel_size=3,
            padding=0,
            sigma_min=0.1,
            sigma_max=2.0,
        ),
        Solarize(prob=0.2, threshold=0.5),
        Clip(0,1),
    ]
})

byolaugmentationdist = [AugmentationDistribution(x) for x in byolaugmentations]



@register(Pipeline, "BYOLOnlinePipeline")
class BYOLOnlinePipeline(Pipeline):
    """
    Augmentations for BYOL.
    From 3.3 of https://arxiv.org/pdf/2006.07733.pdf
    Transformations are:
        - Random crop (p=1.0)
        - Horizontal flip (0.5)
        - Color jitter (0.8)
        - Brightness (0.4)
        - Contrast (0.4)
        - Saturation (0.2)
        - Hue (0.1)
        - Color dropping (0.2)
        - Gaussian blur (T=1.0, T'=0.1)
        - Solarize (T=0.0, T'=0.2)
    Dict of lists of augmentations indexed by view number.

    Args:
       augmentations(Dict[int: List[AugmentationDict]]): list of augmentations
    """

    def __init__(self):
        super().__init__(byolaugmentationdist[0])


@register(Pipeline, "BYOLTargetPipeline")
class BYOLTargetPipeline(Pipeline):
    """
    Augmentations for BYOL.
    From 3.3 of https://arxiv.org/pdf/2006.07733.pdf
    Transformations are:
        - Random crop (p=1.0)
        - Horizontal flip (0.5)
        - Color jitter (0.8)
        - Brightness (0.4)
        - Contrast (0.4)
        - Saturation (0.2)
        - Hue (0.1)
        - Color dropping (0.2)
        - Gaussian blur (T=1.0, T'=0.1)
        - Solarize (T=0.0, T'=0.2)
    Dict of lists of augmentations indexed by view number.

    Args:
       augmentations(Dict[int: List[AugmentationDict]]): list of augmentations
       view(int): which view (set of augmentations)
    """

    def __init__(self):
        super().__init__(byolaugmentationdist[1])
