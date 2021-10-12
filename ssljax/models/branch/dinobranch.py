from ssljax.models.branch.branch import Branch
from ssljax.models.resnet import ResNet50


class DinoOnlineBranch(Branch):
    """
    The online branch in DINO model.
    Rep: Resnet50
    Proj: MLP+softmax
    """

    def setup(self):
        self.resnet50 = Resnet50
        self.mlp = MLP(layer_dims=[200, 10])
        self.softmax = None

    def __call__(self, x):
        pass


class DinoTargetBranch(Branch):
    """
    The target branch in DINO model.
    """

    def setup(self):
        pass

    def __call__(self, x):
        pass
