class TypeCls(object):
    def __init__(self, name, is_pod):
        self.name = name
        self.is_pod = is_pod

Int = TypeCls("int", True) 
int64_t = TypeCls("int64_t", True) 
DataType = TypeCls("DataType", False) 
String = TypeCls("String", False) 
Span = TypeCls("Span", False) 
Type = TypeCls("Type", False) 
