# pylint: disable=no-else-return,
# pylint: disable=unidiomatic-typecheck
"""The set of passes for Relay.

Exposes an interface for configuring the passes and scripting
them in Python.
"""
from . import _ir_pass
# pylint: disable=invalid-name

def infer_type(env, expr):
    """Infer the type of expr under the context of env

    Parameters
    ----------
    env : relay.Environment
        The global environmemt.

    expr : relay.Expr
        The input expression.

    Returns
    -------
    checked_expr : relay.Expr
         The checked expression.
    """
    return _ir_pass.infer_type(env, expr)


well_formed = _ir_pass.well_formed

check_kind = _ir_pass.check_kind

free_vars = _ir_pass.free_vars

free_type_vars = _ir_pass.free_type_vars
