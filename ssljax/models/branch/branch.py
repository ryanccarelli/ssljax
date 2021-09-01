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
