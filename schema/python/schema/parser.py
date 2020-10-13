from .registry import register, lookup
from .expr import ObjectDef, ObjectRefDef, FieldDef
from .codegen import generate
from . import typing as ty

import re
import ast
import inspect


# root class
class Object:
    pass

class ObjectRef:
    pass

_ObjectBaseDef = ObjectDef("Object", "ObjectRef", base=None)
register(_ObjectBaseDef)

_ObjectRefBaseDef = ObjectRefDef("ObjectRef", base=None,
    internal=_ObjectBaseDef)
register(_ObjectRefBaseDef)


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


def parse_comment(text):
    ret = {}
    lines = text.split('\n')
    block_delimiter = [0]
    for lnum, line in enumerate(lines):
        result = re.match(r"\s*-+", line)
        if result:
            block_delimiter.append(lnum - 1)
    block_delimiter.append(len(lines))
    ret['Root'] = lines[block_delimiter[0]: block_delimiter[1]]
    for idx in range(1, len(block_delimiter) - 1):
        key = lines[block_delimiter[idx]].strip()
        ret[key] = lines[block_delimiter[idx]+2: block_delimiter[idx+1]]

    # remove space lines at beginning and the end
    for key in ret:
        lines = ret[key]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
    return ret


def declare(cls):
    print('\n\n')
    name = cls.__name__
    print(name)
    assert issubclass(cls, Object)
    bases = cls.__bases__
    base_name = bases[0].__name__
    base_def = lookup(base_name)

    code = inspect.getsource(cls)
    tree = ast.parse(code)
    parser = Parser()
    parser.visit(tree)

    assert parser.type_key is not None
    type_key = parser.type_key
    fvisit_attrs = parser.default_visit_attrs
    fsequal_reduce = parser.default_sequal_reduce
    fshash_reduce = parser.default_shash_reduce
    fields = parser.fields

    # handle comments
    comment = parse_comment(cls.__doc__)

    obj_def = ObjectDef(name, type_key, base_def, fields,
        fvisit_attrs, fsequal_reduce, fshash_reduce, comment)
    register(obj_def)
    print(generate(obj_def))
    print("{} has been registered.".format(name))

    print('\n')
    base_ref_def = lookup(base_def.type_key)
    objref_def = ObjectRefDef(type_key, base_ref_def, obj_def)
    register(objref_def)
    print(generate(objref_def))
    print("{} has been registered.".format(type_key))

    return cls
