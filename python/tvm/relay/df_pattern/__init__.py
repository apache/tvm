from ....base import Node
from . import _ffi as ffi

def register_df_node(type_key=None):
    """Register a Relay node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return tvm._ffi.register_object(
            "relay.df_pattern" + type_key.__name__)(type_key)
    return tvm._ffi.register_object(type_key)

class DFPattern(Node):
    """Base class of all primitive expressions.

    PrimExpr is used in the low-level code
    optimizations and integer analysis.
    """
    pass

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
