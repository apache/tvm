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
