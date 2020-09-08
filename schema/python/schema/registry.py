from collections import namedtuple
from .expr import ObjectDef, ObjectRefDef
from . import typing as ty

REGISTRY_TABLE = {}

def register(expr):
    assert isinstance(expr, (ObjectDef, ObjectRefDef))
    name = expr.name
    assert name not in REGISTRY_TABLE
    REGISTRY_TABLE[name] = expr
    # register ObjectRef as type
    if isinstance(expr, ObjectRefDef):
        type_ = ty.TypeCls(expr.name, is_pod=False)


def lookup(name):
    if name in REGISTRY_TABLE:
        return REGISTRY_TABLE[name]
    raise ValueError("{} has not been registered.".format(name))
