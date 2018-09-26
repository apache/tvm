# pylint: disable=no-else-return,
# pylint: disable=unidiomatic-typecheck
"""The set of passes for Relay.

Exposes an interface for configuring the passes and scripting
them in Python.
"""
from . import _ir_pass
from . import _make
# pylint: disable=invalid-name

def infer_type(env, expr):
    """Infer the type of expr under the context of env.

    Parameters
    ----------
    env : relay.Environment
        The global environment.

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

def dead_code_elimination(e):
    """ Remove expressions which does not effect the program result (dead code).

    Parameters
    ----------
    e: relay.Expr
      The input Expression

    Returns
    -------
    result: relay.Expr
      An expression which is semantically equal to the input expression,
      but with dead code removed.
    """
    return _ir_pass.dead_code_elimination(e)

def alpha_equal(lhs, rhs):
    """Compare two Relay expr for structural equivalence (alpha equivalence).

    Parameters
    ----------
    lhs: relay.Expr
      One of the input Expression.
    rhs: relay.Expr
      One of the input Expression.


    Returns
    -------
    result: bool
      True iff lhs is alpha equal to rhs.
    """
    return bool(_make._alpha_equal(lhs, rhs))
