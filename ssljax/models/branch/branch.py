from ssljax.models.model import Model


class Branch(Model):
    """
    A branch is a nn.Module that is executed in parallel with other branches.
    Branches execute first a model body (eg. ResNet, ViT), then optionally
    a model head and predictor (eg. MLP).
    """

    def setup(self, **args):
        pass
