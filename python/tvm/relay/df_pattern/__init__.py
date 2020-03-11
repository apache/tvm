from ...ir.base import Node
from ...ir import make_node
from ... import _ffi as tvm_ffi
from ..op import get
from . import _ffi as ffi

def register_df_node(type_key=None):
    """Register a Relay node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return tvm_ffi.register_object(
            "relay.df_pattern." + type_key.__name__)(type_key)
    return tvm_ffi.register_object(type_key)

class DFPattern(Node):
    """Base class of all primitive expressions.

    PrimExpr is used in the low-level code
    optimizations and integer analysis.
    """
    def __call__(self, *args):
        return CallPattern(self, list(args))

    def __or__(self, other):
        return AltPattern(self, other)

    def has_attr(self, attr_name, attr_value):
        attrs = make_node("DictAttrs", **{attr_name:attr_value})
        return AttrPattern(self, attrs)

@register_df_node
class ExprPattern(DFPattern):
    """A pattern which matches a constant expression.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The expression to match.
    """
    def __init__(self, expr):
        self.__init_handle_by_constructor__(ffi.ExprPattern, expr)

@register_df_node
class VarPattern(DFPattern):
    """A local variable in Relay.

    Local variable can be used to declare input
    arguments to a function, or intermediate variables.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: tvm.relay.Type, optional
        The type annotation on the variable.
    """
    def __init__(self, name_hint, type_annotation=None):
        self.__init_handle_by_constructor__(
            ffi.VarPattern, name_hint, type_annotation)

#     @property
#     def name_hint(self):
#         """Get name hint of the current var."""
#         name = self.name
#         return name


@register_df_node
class CallPattern(DFPattern):
    """A pattern matching a function call node in Relay.

    Parameters
    ----------
    op: realy.df_pattern.DFPattern
        The operation to be called.

    args: List[realy.df_pattern.DFPattern]
        The arguments to the call.

    attrs: Optional[tvm.Attrs]
        Attributes to the call, can be None

    type_args: Optional[List[tvm.relay.Type]]
        The additional type arguments, this is only
        used in advanced usecase of template functions.
    """
    def __init__(self, op, args, attrs=None, type_args=None):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(
            ffi.CallPattern, op, args, attrs, type_args)

@register_df_node
class TuplePattern(DFPattern):
    """A patern matching a Relay Tuple.

    Parameters
    ----------
    fields : List[tvm.relay.df_pattern.DFPattern]
        The fields in the tuple.
    """
    def __init__(self, fields):
        self.__init_handle_by_constructor__(ffi.TuplePattern, fields)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("TuplePattern index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on TuplePattern")

@register_df_node
class TupleGetItemPattern(DFPattern):
    """Get index-th item from a TuplePattern.

    Parameters
    ----------
    tuple_value: tvm.relay.df_pattern.DFPattern
        The input tuple expression.

    index: int
        The index.
    """
    def __init__(self, tuple_value, index):
        self.__init_handle_by_constructor__(
            ffi.TupleGetItemPattern, tuple_value, index)

@register_df_node
class AltPattern(DFPattern):
    """Create a Pattern that can match one of two conditions

    Parameters
    ----------
    left: tvm.relay.df_pattern.DFPattern
        One possible matching Pattern
    right: tvm.relay.df_pattern.DFPattern
        One possible matching Pattern
    """
    def __init__(self, tuple_value, index):
        self.__init_handle_by_constructor__(
            ffi.AltPattern, tuple_value, index)

@register_df_node
class WildcardPattern(DFPattern):
    """A pattern which matches anything.
    """
    def __init__(self):
        self.__init_handle_by_constructor__(ffi.WildcardPattern)

@register_df_node
class TypePattern(DFPattern):
    """Get index-th item from a TuplePattern.

    Parameters
    ----------
    pattern: tvm.relay.df_pattern.DFPattern
        The input tuple expression.

    ttype: tvm.relay.Type
        The type to match
    """
    def __init__(self, pattern, ttype):
        self.__init_handle_by_constructor__(
            ffi.TypePattern, pattern, ttype)

@register_df_node
class AttrPattern(DFPattern):
    """Get index-th item from a TuplePattern.

    Parameters
    ----------
    pattern: tvm.relay.df_pattern.DFPattern
        The input tuple expression.

    attrs: tvm.Attrs
        The attributes to match
    """
    def __init__(self, pattern, attrs):
        self.__init_handle_by_constructor__(
            ffi.AttrPattern, pattern, attrs)

def is_input(name=None) -> DFPattern:
    return VarPattern(name)

def is_op(op_name: str) -> DFPattern:
    op = get(op_name)
    return ExprPattern(op)

def wildcard() -> DFPattern:
    return WildcardPattern()

def has_type(ty, pattern=None):
    if pattern is None:
        pattern = wildcard()
    return TypePattern(pattern, ty)

def has_attr(attr_name, attr_value, pattern=None):
    if pattern is None:
        pattern = wildcard()
    return patter.has_attr(attr_name, attr_value)
