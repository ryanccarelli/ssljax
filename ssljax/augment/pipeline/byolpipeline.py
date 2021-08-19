import jax
import jax.numpy as jnp


class BYOLPipeline(Pipeline):
    """
    Augmentations for BYOL and SimCLR.
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
    """

    def __init__(self, augmentprobs=[], rng):
        self.rng = rng
        if isinstance(augmentprobs, dict):
            return self.sample(self.rng)
        elif isinstance(augmentprobs, list):
            super().__init__(augmentlist)

    @classmethod
    def sample(cls, rng):
        flip_rng, color_rng, blur_rng, solarize_rng = jax.random.split(rng, 4)

