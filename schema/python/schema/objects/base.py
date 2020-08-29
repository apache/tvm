from .. import typing as ty
from ..parser import declare, Object, ObjectRef

@declare
class BaseExprNode(Object): 
    type_key = "BaseExpr"

@declare
class BaseExpr(ObjectRef):
    internal_object = BaseExprNode

@declare
class PrimExprNode(BaseExprNode):
    type_key = "PrimExpr"
    def __init__(self, dtype: ty.DataType):
        pass
    
@declare
class PrimExpr(BaseExpr):
    internal_object = PrimExprNode

@declare
class RelayExprNode(BaseExprNode):
    type_key = "RelayExpr"
    def __init__(self, span: ty.Span, checked_type_: ty.Type):
        pass

@declare
class RelayExpr(BaseExpr):
    internal_object = RelayExprNode

@declare
class GlobalVarNode(RelayExprNode):
    type_key = "GlobalVar"
    def __init__(self, name_hint: ty.String):
        pass

    def VisitAttrs(self):
        pass

@declare
class GlobalVar(RelayExpr):
    internal_object = GlobalVarNode

@declare
class IntImmNode(PrimExprNode):
    type_key = "IntImm"
    def __init__(self, value: ty.int64_t):
        pass

    def VisitAttrs(self):
        pass

    def SEqualReduce(self):
        pass

    def SHashReduce(self):
        pass

@declare
class IntImm(PrimExpr):
    internal_object = IntImmNode
