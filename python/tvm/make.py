"""namespace of IR node builder make function

This namespace is used for developers. While you do not see any declarations.
The functions are automatically exported from C++ side via PackedFunc.

Each api is a PackedFunc that can be called in a positional argument manner.
You can use make function to build the IR node.
"""
from ._ffi.function import _init_api
from . import stmt as _stmt

def range_by_min_extent(min_value, extent):
    """Construct a Range by min and extent.

    This constructs a range in [min_value, min_value + extent)

    Parameters
    ----------
    min_value : Expr
        The minimum value of the range.

    extent : Expr
        The extent of the range.

    Returns
    -------
    rng : Range
        The constructed range.
    """
    return _range_by_min_extent(min_value, extent)


def node(type_key, **kwargs):
    """Make a new DSL node by its type key and fields

    Parameters
    ----------
    type_key : str
        The type key of the node.

    **kwargs : dict
        The fields of the node.

    Example
    -------
    The following code constructs a IntImm object

    .. code-block:: python

       x = tvm.make.node("IntImm", dtype="int32", value=10)
       assert isinstance(x, tvm.expr.IntImm)
       assert x.value == 10
    """
    args = [type_key]
    for k, v in kwargs.items():
        args += [k, v]
    return _Node(*args)


def stmt_seq(*args):
    """Make sequence of statements

    Parameters
    ----------
    args : list of Expr or Var
        List of statements to be combined as sequence.

    Returns
    -------
    stmt : Stmt
        The combined statement.
    """
    ret = None
    for value in args:
        if not isinstance(value, _stmt.Stmt):
            value = Evaluate(value)
        ret = value if ret is None else Block(ret, value)
    return ret if ret else Evaluate(0)

_init_api("tvm.make")
