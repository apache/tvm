from collections import namedtuple
from .expr import ObjectDef, ObjectRefDef

ExprInfo = namedtuple('ExprInfo', ['name', 'schema_type'])

REGISTRY_TABLE = {}
STR2TYPE = {
    'ObjectDef': ObjectDef,
    'ObjectRefDef': ObjectRefDef,
}

def register(expr):
    assert isinstance(expr, (ObjectDef, ObjectRefDef))
    name = expr.name
    schema_type = type(expr)
    info = ExprInfo(name, schema_type)
    assert info not in REGISTRY_TABLE
    REGISTRY_TABLE[info] = expr

def lookup(name, schema_type):
    assert isinstance(schema_type, str)
    schema_type = STR2TYPE[schema_type]
    info = ExprInfo(name, schema_type)
    if info in REGISTRY_TABLE:
        return REGISTRY_TABLE[info]
    raise ValueError("{}:{} has not been registered.".format(name, schema_type))
