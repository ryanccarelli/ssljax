from collections import OrderedDict

import jax
import jax.numpy as jnp
from ssljax.augment.pipeline.pipeline import Pipeline

# TODO: clip is followed by stop_grad
byolaugmentations = collections.OrderedDict({0: {[
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
            blur_divider=10.,
            sigma_min=0.1,
            sigma_max=2.0,
        ),
        Clip(0,1),
    ]},
    1: {[
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
            blur_divider=10.,
            sigma_min=0.1,
            sigma_max=2.0,
        ),
        Solarize(prob=0.2, threshold=0.5),
        Clip(0,1),
    ]}
})
byolaugmentationdist = [AugmentationDistribution(x) for x in byolaugmentations]
class BYOLPipeline(Pipeline):
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

    def __init__(self, augmentations=byolaugmentationsdict, view):
        super().__init__(augmentations[view])
