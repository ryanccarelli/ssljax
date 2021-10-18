import flax.linen as nn
import jax.lax
from ssljax.core.utils import get_from_register, register
from ssljax.models.model import Model


class Branch(Model):
    """
    A branch is a nn.Module that is executed in parallel with other branches.
    Branches execute first a model body (eg. ResNet, ViT), then optionally
    a model head and predictor (eg. MLP).
    """

    def setup(self, **args):
        pass


@register(Branch, "OnlineBranch")
class OnlineBranch(Branch):
    """
    An online branch for a self-supervised learning model.

    Args:
        body (str): body to retrieve from register, must index Model
        head (str): head to retrieve from register, must index Model
        pred (str): predictor to retrieve from register, must index Model
    """

    body: dict
    head: dict
    pred: dict

    def setup(self):
        self.body_mod = get_from_register(Model, self.body.name)(**self.body.params)
        self.head_mod = get_from_register(Model, self.head.name)(**self.head.params)
        self.pred_mod = get_from_register(Model, self.pred.name)(**self.pred.params)

    def __call__(self, x):
        x = self.body_mod(x)
        x = self.head_mod(x)
        x = self.pred_mod(x)
        return x


@register(Branch, "TargetBranch")
class TargetBranch(Branch):
    """
    A target branch for a self-supervised learning model.

    Args:
        body (str): body to retrieve from register, must index Model
        head (str): head to retrieve from register, must index Model
    """

    body: dict
    head: dict

    def setup(self):
        self.body_mod = get_from_register(Model, self.body.name)(**self.body.params)
        self.head_mod = get_from_register(Model, self.head.name)(**self.head.params)

    def __call__(self, x):
        x = self.body_mod(x)
        x = self.head_mod(x)
        return jax.lax.stop_gradient(x)
