import flax.linen as nn
import jax.lax
from ssljax.core.utils import get_from_register, register
from ssljax.models.model import Model


class Branch(Model):
    """
    Base class for a ssljax branch.
    A branch is a model body and (optionally) head that will
    be executed in parallel with other branches.

    Through setup, a branch defines the keys that index
    its parameter groups. For example,
    > self.linear = nn.Dense(10)
    > self.linear2 = nn.Dense(20)
    results in a parameter dict {linear: Params, linear2: Params}.
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

    body_params: dict
    head_params: dict
    pred_params: dict

    def setup(self):
        print("body params jer", self.body_params)
        self.body = get_from_register(Model, self.body_params.name)(**self.body_params.params)
        self.head = get_from_register(Model, self.head_params.name)(**self.head_params.params)
        self.pred = get_from_register(Model, self.pred_params.name)(**self.pred_params.params)

    def __call__(self, x):
        x = self.body(x)
        x = self.head(x)
        x = self.pred(x)
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
        self.body = get_from_register(Model, self.body.name)(**self.body.params)
        self.head = get_from_register(Model, self.head.name)(**self.head.params)

    def __call__(self, x):
        x = self.body(x)
        x = self.head(x)
        return jax.lax.stop_gradient(x)
