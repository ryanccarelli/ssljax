# branches for byol
from ssljax.models.branch.branch import Branch
from ssljax.models.resnet import ResNet50


@register(Branch, "BYOLOnlineBranch")
class BYOLOnlineBranch(Branch):
    """
    The online branch for a BYOl model.
    See: https://arxiv.org/abs/2006.07733
    """

    def setup(self):
        self.resnet50 = Resnet50
        self.mlp = MLP(layer_dims=[200, 10])

    def __call__(self, x):
        x = self.resnet50(x)
        x = self.mlp(x)
        return x


@register(Branch, "BYOLTargetBranch")
class BYOLTargetBranch(Branch):
    """
    The target branch for a BYOl model.
    See: https://arxiv.org/abs/2006.07733
    """

    def setup(self):
        self.resnet50 = Resnet50

    def __call__(self, x):
        x = self.resnet50(x)
        return x
