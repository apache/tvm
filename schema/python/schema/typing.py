_GLOBAL_TYPE_TABLE = {}

def get_type(name):
    assert name in _GLOBAL_TYPE_TABLE
    return _GLOBAL_TYPE_TABLE[name]

class TypeCls(object):
    def __init__(self, name, is_pod):
        self.name = name
        self.is_pod = is_pod
        _GLOBAL_TYPE_TABLE[name] = self

Int = TypeCls("int", True) 
int64_t = TypeCls("int64_t", True) 
DataType = TypeCls("DataType", False) 
String = TypeCls("String", False) 
Span = TypeCls("Span", False) 
Type = TypeCls("Type", False) 
