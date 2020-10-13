from .. import typing as ty
from ..parser import declare, Object, ObjectRef


@declare
class BaseExprNode(Object): 
    """
    Base type of all the expression.

    See Also
    --------
    BaseExpr
    """
    type_key = "BaseExpr"
    default_visit_attrs = False
    default_sequal_reduce = False
    default_shash_reduce = False


@declare
class PrimExprNode(BaseExprNode):
    """
    Base node of all primitive expressions.
 
    A primitive expression deals with low-level
    POD data types and handles without
    doing life-cycle management for objects.
 
    PrimExpr is used in the low-level code
    optimizations and integer analysis.

    See Also 
    --------
    PrimExpr

    Attributes 
    ---------- 
    dtype 
        The The runtime data type of the primitive expression.

        runtime::DataType(dtype) provides coarse grained type information
        during compile time and runtime. It is eagerly built in
        PrimExpr expression construction and can be used for
        quick type checking.
        
        dtype is sufficient to decide the Type of the PrimExpr
        when it corresponds to POD value types such as i32.
        
        When dtype is DataType::Handle(), the expression could corresponds to
        a more fine-grained Type, and we can get the type by running lazy type inference.
    """
    type_key = "PrimExpr"
    default_visit_attrs = False
    default_sequal_reduce = False
    default_shash_reduce = False
    dtype: ty.DataType
    

@declare
class RelayExprNode(BaseExprNode):
    """
    Base node of all non-primitive expressions.

    RelayExpr supports tensor types, functions and ADT as
    first class citizens. The life-cycle of the corresponding
    objects are implicitly managed by the language.
    
    See Also 
    --------
    RelayExpr

    Attributes 
    ----------
    span 
        Span that points to the original source code.
        Reserved debug information.
    checked_type_ 
        Stores the result of type inference(type checking).
    """
    type_key = "RelayExpr"
    default_visit_attrs = False
    default_sequal_reduce = False
    default_shash_reduce = False
    span: ty.Span
    checked_type_: ty.Type


@declare
class GlobalVarNode(RelayExprNode):
    """
    Global variable that lives in the top-level module.
    
    A GlobalVar only refers to function definitions.
    This is used to enable recursive calls between function.
    
    See Also
    --------
    GlobalVarNode

    Attributes
    ----------
    name_hint
        The name of the variable, this only acts as a hint.
    """
    type_key = "GlobalVar"
    default_sequal_reduce = False
    default_shash_reduce = False
    name_hint: ty.String


@declare
class IntImmNode(PrimExprNode):
    """
    Constant integer literals in the program.

    See Also
    --------
    IntImm

    Attributes
    ----------
    value
        The internal value.
    """
    type_key = "IntImm"
    value: ty.int64_t
