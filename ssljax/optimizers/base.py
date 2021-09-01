from typing import Any, Callable, NamedTuple, Optional, Tuple

from optax._src import base

ParameterUpdateFn = Callable[
    [base.OptState, base.Params], Tuple[base.Params, base.OptState]
]


class ParameterTransformation(NamedTuple):
    """Optax transformation concisting of a function pair: (initialise, update)."""

    init: base.TransformInitFn
    update: ParameterUpdateFn
