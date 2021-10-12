import logging

logger = logging.getLogger(__name__)


def wrap_func_in_class_call(name, func, BaseClass=object):
    """ Wraps a function in a class with the function as __call__

    Args:
        name(str): Name of the new class
        func(Callable): A function which will be the mapped to __call__
        BaseClass(Class, <optional>): Base class of the new function
    """
    newclass = type(name, (BaseClass,), {"__call__": lambda self, *args: func(*args)})
    return newclass
