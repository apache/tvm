from pprint import pprint

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


class ObjectRefDef(SchemaExpr):
    def __init__(self, name, base, internal):
        self.name = name 
        self.base = base
        self.internal = internal

class FieldDef(SchemaExpr):
    def __init__(self, name, type_):
        self.name = name
        self.type_ = type_

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(vars(self))
