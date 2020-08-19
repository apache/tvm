class SchemaExpr(object):
    pass

class ObjectDef(SchemaExpr):
    def __init__(self, name, type_key, base, fields=[],
                 fvisit_attrs=False, fsequal_reduce=False,
                 fshash_reduce=False):
        self.name = name
        self.type_key = type_key
        self.base = base
        self.fields = fields
        self.fvisit_attrs = fvisit_attrs
        self.fsequal_reduce = fsequal_reduce
        self.fshash_reduce = fshash_reduce

ObjectBase = ObjectDef("Object", "Object", base=None)

class ObjectRefDef(SchemaExpr):
    def __init__(self, name, base, internal_object):
        self.name = name 
        self.base = base
        self.internal_object = internal_object

ObjectRefBase = ObjectRefDef("ObjectRef", base=None, internal_object=ObjectBase)

class FieldDef(SchemaExpr):
    def __init__(self, name, type_):
        self.name = name
        self.type_ = type_
