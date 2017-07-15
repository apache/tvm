"""namespace of IR node builder make function

This namespace is used for developers. While you do not see any declarations.
The functions are automatically exported from C++ side via PackedFunc.

Each api is a PackedFunc that can be called in a positional argument manner.
You can use make function to build the IR node.
"""
from ._ffi.function import _init_api

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

_init_api("tvm.make")
