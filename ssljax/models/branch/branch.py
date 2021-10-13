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

    def setup(self, body, body_params, head, head_params, pred, pred_params):
        self.body = get_from_register(Model, body, **body_params)
        self.head = get_from_register(Model, head, **head_params)
        self.predictor = get_from_register(Model, pred, **pred_params)

    def __call__(self):
        x = self.body(x)
        x = self.head(x)
        x = self.predictor(x)
        return x


@register(Branch, "TargetBranch")
class TargetBranch(Branch):
    """
    A target branch for a self-supervised learning model.

    Args:
        body (str): body to retrieve from register, must index Model
        head (str): head to retrieve from register, must index Model
    """

    def setup(self, body, body_params, head, head_params, pred, pred_params):
        self.body = get_from_register(Model, body, **body_params)
        self.head = get_from_register(Model, head, **head_params)

    def __call__(self):
        x = self.body(x)
        x = self.head(x)
        return x
