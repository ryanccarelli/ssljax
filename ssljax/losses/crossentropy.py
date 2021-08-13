from ssljax.core.utils import Registrable


@Loss.register("crossentropy")
class CrossEntropy:
    """
    Generalized cross-entropy loss.
    For a single tensor this is cross entropy, for a list of tensors
    this is the sum of the cross-entropy losses for each tensor in the list.
    """
    def __init__(self):
        
