class Type(object):
    def __init__(self, name, is_pod):
        self.name = name
        self.is_pod = is_pod

Int = Type("int", True) 
DataType = Type("DataType", False) 
String = Type("String", False) 
Span = Type("Span", False) 
Type = Type("Type", False) 

class SchemaExpr(object):
    pass

class ObjectDef(SchemaExpr):
    def __init__(self, name, type_key, base, fields=[], fvisit_attrs=False):
        self.name = name
        self.type_key = type_key
        self.base = base
        self.fields = fields
        self.fvisit_attrs = fvisit_attrs

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

