from collections import namedtuple
from .expr import ObjectDef, ObjectRefDef

REGISTRY_TABLE = {}
# STR2TYPE = {
#     'ObjectDef': ObjectDef,
#     'ObjectRefDef': ObjectRefDef,
# }

def register(expr):
    assert isinstance(expr, (ObjectDef, ObjectRefDef))
    name = expr.name
    assert name not in REGISTRY_TABLE
    REGISTRY_TABLE[name] = expr

def lookup(name):
    if name in REGISTRY_TABLE:
        return REGISTRY_TABLE[name]
    raise ValueError("{} has not been registered.".format(name))
