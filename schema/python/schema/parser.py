from .registry import register, lookup
from .expr import ObjectDef, ObjectRefDef, FieldDef
from .codegen import generate

import inspect


class Object(object):
    type_key = "Object"

_ObjectBaseDef = ObjectDef("Object", "Object", base=None)
register(_ObjectBaseDef)


class ObjectRef:
    internal_object = Object

_ObjectRefBaseDef = ObjectRefDef("ObjectRef", base=None,
    internal_object=_ObjectBaseDef)
register(_ObjectRefBaseDef)


def declare(cls):
    print('\n\n')
    name = cls.__name__
    print(name)
    bases = cls.__bases__
    base_name = bases[0].__name__
    base_def = lookup(base_name)

    if issubclass(cls, Object):
        type_key = cls.type_key
        # default
        fields = []
        fvisit_attrs = False
        fsequal_reduce = False
        fshash_reduce = False

        field_spec = inspect.getfullargspec(cls.__init__)
        for field_name in field_spec.args[1:]:
            field_type = field_spec.annotations[field_name]
            fields.append(FieldDef(field_name, field_type))

        if hasattr(cls, "VisitAttrs"):
            fvisit_attrs = True
        if hasattr(cls, "SHashReduce"):
            fshash_reduce = True
        if hasattr(cls, "SEqualReduce"):
            fsequal_reduce = True

        def_ = ObjectDef(name, type_key, base_def, fields,
            fvisit_attrs, fsequal_reduce, fshash_reduce)
        register(def_)
        print(generate(def_))
        print("{} has been registered.".format(name))

    if issubclass(cls, ObjectRef):
        object_class = cls.internal_object
        internal_object = lookup(object_class.__name__)
        def_ = ObjectRefDef(name, base_def, internal_object)
        register(def_)
        print(generate(def_))
        print("{} has been registered.".format(name))

    return cls
