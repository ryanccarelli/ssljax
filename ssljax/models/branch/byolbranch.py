# branches for byol
from ssljax.models.branch import Branch
from ssljax.models.resnet import Resnet50


@register(Branch, "BYOLOnlineBranch")
class BYOLOnlineBranch(Branch):
    """
    The online branch in BYOL model.
    Rep: Resnet50
    Proj: MLP
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
    The target branch in BYOL model.
    Rep: Resnet50
    """
    def setup(self):
        self.resnet50 = Resnet50

    def __call__(self):
        x = self.resnet50(x)
        return x
