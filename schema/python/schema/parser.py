from .registry import register, lookup
from .expr import ObjectDef, ObjectRefDef, FieldDef
from .codegen import generate
from . import typing as ty

import inspect
import ast


class Parser(ast.NodeVisitor):
    def __init__(self):
        self.type_key = None
        self.default_visit_attrs = True
        self.default_sequal_reduce = True
        self.default_shash_reduce = True
        self.fields = []

    def visit_Assign(self, node):
        target = node.targets[0]
        value = node.value
        if target.id == 'type_key':
            self.type_key = value.s
        elif target.id == 'default_visit_attrs':
            self.default_visit_attrs = value.value
        elif target.id == 'default_sequal_reduce':
            self.default_sequal_reduce = value.value
        elif target.id == 'default_shash_reduce':
            self.default_shash_reduce = value.value
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        name = node.target.id
        type_name = node.annotation.attr
        type_ = ty.get_type(type_name)
        self.fields.append(FieldDef(name, type_))
        self.generic_visit(node)


# root class
class Object:
    pass

class ObjectRef:
    pass

_ObjectBaseDef = ObjectDef("Object", "Object", base=None)
register(_ObjectBaseDef)

_ObjectRefBaseDef = ObjectRefDef("ObjectRef", base=None,
    internal=_ObjectBaseDef)
register(_ObjectRefBaseDef)


def declare(cls):
    print('\n\n')
    name = cls.__name__
    print(name)
    bases = cls.__bases__
    base_name = bases[0].__name__
    base_def = lookup(base_name)

    code = inspect.getsource(cls)
    tree = ast.parse(code)
    parser = Parser()
    parser.visit(tree)

    if issubclass(cls, Object):
        assert parser.type_key is not None
        type_key = parser.type_key
        fvisit_attrs = parser.default_visit_attrs
        fsequal_reduce = parser.default_sequal_reduce
        fshash_reduce = parser.default_shash_reduce
        fields = parser.fields

        def_ = ObjectDef(name, type_key, base_def, fields,
            fvisit_attrs, fsequal_reduce, fshash_reduce)
        register(def_)
        print(generate(def_))
        print("{} has been registered.".format(name))

    if issubclass(cls, ObjectRef):
        object_class = cls.internal
        internal = lookup(object_class.__name__)
        def_ = ObjectRefDef(name, base_def, internal)
        register(def_)
        print(generate(def_))
        print("{} has been registered.".format(name))

    return cls
