# pylint: disable=no-else-return,
# pylint: disable=unidiomatic-typecheck
"""The set of passes for Relay.

Exposes an interface for configuring the passes and
scripting them in Python.
"""
from . import _ir_pass
from . import _make
# pylint: disable=invalid-name

def infer_type(expr, env=None):
    """Infer the type of expr under the context of env.

    Parameters
    ----------
    expr: tvm.relay.Expr
      The input expression.

    env: Optional[tvm.relay.Environment]
      The global environment.


    Returns
    -------
    checked_expr : tvm.relay.Expr
      The checked expression.
    """
    return _ir_pass.infer_type(expr, env)


def well_formed(expr):
    """Check that each Var is only bound once (well formed).

    Parameters
    ----------
    expr: tvm.relay.Expr
      The input expression

    Returns
    -------
    well_form : bool
      whether the input expression is well formed
    """
    return _ir_pass.well_formed(expr)


def check_kind(t, env=None):
    """Check that the type is well kinded.
    For example, this mean type cannot has tensor of tensor, or is a tuple type of 2 shapes.

    Parameters
    ----------
    t: tvm.relay.Type
      The type to check

    env: tvm.relay.Environment, optional
      The global environment

    Returns
    -------
    well_kinded : bool
      whether the input type is well kinded.

    Examples
    --------
    .. code:: python

        assert not check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Shape)]))
        assert check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Type)]))
    """
    if env is not None:
        return _ir_pass.check_kind(t, env)
    else:
        return _ir_pass.check_kind(t)


def free_vars(e):
    """Get free variables from expression e.

    Parameters
    ----------
    e: tvm.relay.Expr
      The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of free variables
    """
    return _ir_pass.free_vars(e)


def free_type_vars(expr):
    """Get free type variables from expression/type e

    Parameters
    ----------
    expr: Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type

    Returns
    -------
    free : List[tvm.relay.TypeParam]
        The list of free type variables
    """
    return _ir_pass.free_type_vars(expr)


def dead_code_elimination(expr):
    """ Remove expressions which does not effect the program result (dead code).

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
        An expression which is semantically equal to the input expression,
        but with dead code removed.
    """
    return _ir_pass.dead_code_elimination(expr)


def alpha_equal(lhs, rhs):
    """Compare two Relay expr for structural equivalence (alpha equivalence).

    Parameters
    ----------
    lhs: tvm.relay.Expr
      One of the input Expression.

    rhs: tvm.relay.Expr
      One of the input Expression.

    Returns
    -------
    result: bool
      True iff lhs is alpha equal to rhs.
    """
    return bool(_make._alpha_equal(lhs, rhs))
