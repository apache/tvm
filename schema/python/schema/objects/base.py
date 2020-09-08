from .. import typing as ty
from ..parser import declare, Object, ObjectRef


@declare
class BaseExprNode(Object): 
    type_key = "BaseExpr"
    default_visit_attrs = False
    default_sequal_reduce = False
    default_shash_reduce = False

@declare
class BaseExpr(ObjectRef):
    internal = BaseExprNode

@declare
class PrimExprNode(BaseExprNode):
    type_key = "PrimExpr"
    default_visit_attrs = False
    default_sequal_reduce = False
    default_shash_reduce = False
    dtype: ty.DataType
    
@declare
class PrimExpr(BaseExpr):
    internal = PrimExprNode

@declare
class RelayExprNode(BaseExprNode):
    type_key = "RelayExpr"
    default_visit_attrs = False
    default_sequal_reduce = False
    default_shash_reduce = False
    span: ty.Span
    checked_type_: ty.Type

@declare
class RelayExpr(BaseExpr):
    internal = RelayExprNode

@declare
class GlobalVarNode(RelayExprNode):
    type_key = "GlobalVar"
    default_sequal_reduce = False
    default_shash_reduce = False
    name_hint: ty.String

@declare
class GlobalVar(RelayExpr):
    internal = GlobalVarNode

@declare
class IntImmNode(PrimExprNode):
    type_key = "IntImm"
    value: ty.int64_t

@declare
class IntImm(PrimExpr):
    internal = IntImmNode
