# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
The set of automatic differentiation algorithms in Relay.
"""
from . import _gradient

def gradient(expr, mod=None, order=1):
    if order == 1:
        return _gradient.first_order_gradient(expr, mod)
    else:
        raise Exception('only order=1 supported right now.')
