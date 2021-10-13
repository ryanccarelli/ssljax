import collections
import inspect
import pprint

GLOBAL_TYPE_REGISTRY = collections.defaultdict(dict)


def print_registry():
    """
    Pretty print the registry

    Args: None
    """
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(GLOBAL_TYPE_REGISTRY)


def register(type_, name, override=False):
    """
    Register is a decorator to register based on a type and name.

    Args:
        type (class/callable): Specify the type of the class/module being registered.
        name (str): Specify the name of the entity that is being registered.
        override(bool, optional): Disable uniqueness check on (name, type) in the
            GLOBAL_TYPE_REGISTRY

    Example::

        class ModelBase:
            pass

        @register(ModelBase, "CNN", override=True)
        class CNN(nn.Module, ModelBase):

          @nn.compact
          def __call__(self, x):
              # Implementation here

    """

    def register_entity(entity):
        if name in GLOBAL_TYPE_REGISTRY[type_] and not override:
            msg = "Cannot register duplicate model ({}). Already registered in GLOBAL_TYPE_REGISTRY"
            raise ValueError(msg.format(name, type_))
        if inspect.isclass(entity) and not issubclass(entity, type_):
            raise ValueError(
                "Entity ({}: {}) must extend {}".format(name, entity.__name__, type_)
            )
        if not callable(entity):
            raise ValueError(
                "Entity  doesn't match input type nor is a callable ({}: {})".format(
                    name, entity.__name__
                )
            )

        GLOBAL_TYPE_REGISTRY[type_][name] = entity
        return entity

    return register_entity


def get_from_register(type_, name):
    """
    get_from_register gets the entity (class or function) from the global
    registry if it exists.  Else, throws a ValueError

    Args;
        type_ (class/callable): Specify the type of the class/module being retrieved
        name (str): Specify the name of the entity that is being retrieved.

    Returns:
        entity (class/callable): Returns the entity from the global register.
            Throws ValueError if not found.
    """
    if type_ not in GLOBAL_TYPE_REGISTRY:
        msg = "No type {} in global registry"
        raise ValueError(msg.format(type_))
    if name not in GLOBAL_TYPE_REGISTRY[type_]:
        msg = "No entity {} registered in global registry with type {}"
        raise ValueError(msg.format(name, type_))

    return GLOBAL_TYPE_REGISTRY[type_][name]

