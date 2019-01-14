# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
The set of automatic differentiation algorithms in Relay.
"""
from . import _gradient


def gradient(expr, mod=None):
    """.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, which is a Function or a GlobalVar.

    mod : Optional[tvm.relay.Module]
        The global module.

    Returns
    -------
    ret : tvm.relay.Expr
        A function that calculate the original result paired with gradient.
    """
    return _gradient.first_order_gradient(expr, mod)
