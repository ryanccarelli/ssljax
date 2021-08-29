# branches for byol
from ssljax.models.branch import Branch


class ByolOnlineBranch(Branch):
    """
    The online branch in BYOL model.
    Rep: Resnet50
    Proj: MLP
    Pred: MLP
    """

    def __call__(self):
        pass


class ByolTargetBranch(Branch):
    """
    The target branch in BYOL model.
    Rep: Resnet50
    Proj: MLP
    """

    def __call__(self):
        pass
