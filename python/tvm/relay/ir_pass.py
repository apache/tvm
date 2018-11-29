# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""The set of passes for Relay.

Exposes an interface for configuring the passes and
scripting them in Python.
"""
from . import _ir_pass
from . import _make
from .expr import Expr
from .ty import Type

def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.
    fvisit : function
        The visitor function to be applied.
    """
    return _ir_pass.post_order_visit(expr, fvisit)

def infer_type(expr, mod=None):
    """Infer the type of expr under the context of mod.

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression.

    mod: Optional[tvm.relay.Module]
        The global module.


    Returns
    -------
    checked_expr : tvm.relay.Expr
        The checked expression.
    """
    return _ir_pass.infer_type(expr, mod)


def backward_fold_scale_axis(expr):
    """Backward fold axis scaling into weights of conv2d/dense.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, we expect that expr's types
        should be fully inferred by infer_type.

    Returns
    -------
    folded_expr : tvm.relay.Expr
        The folded expression after transformation.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis.
    As backward folding targets common conv-bn pattern.
    """
    return _ir_pass.backward_fold_scale_axis(expr)


def forward_fold_scale_axis(expr):
    """Fold the scaling of axis into weights of conv2d/dense.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression, we expect that expr's types
        should be fully inferred by infer_type.

    Returns
    -------
    folded_expr : tvm.relay.Expr
        The folded expression after transformation.

    Note
    ----
    It is recommended to call backward_fold_scale_axis
    before using forward_fold_scale_axis.
    As backward folding targets common conv-bn pattern.
    """
    return _ir_pass.forward_fold_scale_axis(expr)


def well_formed(expr):
    """Check that each Var is only bound once (well formed).

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression

    Returns
    -------
    well_form : bool
        Whether the input expression is well formed
    """
    return _ir_pass.well_formed(expr)


def check_kind(t, mod=None):
    """Check that the type is well kinded.
    For example, this mean type cannot has tensor of tensor, or is a tuple type of 2 shapes.

    Parameters
    ----------
    t: tvm.relay.Type
        The type to check

    mod: tvm.relay.Module, optional
        The global module

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
    if mod is not None:
        return _ir_pass.check_kind(t, mod)
    else:
        return _ir_pass.check_kind(t)


def free_vars(expr):
    """Get free Vars from expression expr in Post DFS order.

    Parameters
    ----------
    expr: tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of free variables in post DFS order.

    Note
    ----
    The fact that Vars are post-DFS ordred are useful in
    neural networks: usually this means weights of previous
    are ordered first.
    """
    return _ir_pass.free_vars(expr)


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


def simplify_inference(expr):
    """ Simplify the data-flow graph for inference phase.

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
        An expression which is semantically equal to the input expression,
        but with some simplification
    """
    return _ir_pass.simplify_inference(expr)


def canonicalize_ops(expr):
    """ Canonicalize special operators to basic operators.
    This can simplify latter analysis. (e.g. Expand bias_add to expand_dims and broadcast_add.)

    Parameters
    ----------
    e: tvm.relay.Expr
        The input Expression

    Returns
    -------
    result: tvm.relay.Expr
        An expression without bias_add
    """
    return _ir_pass.canonicalize_ops(expr)


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


def graph_equal(lhs, rhs):
    """Compare two Relay expr for data-flow equivalence.
    The difference between this and alpha-equality is that
    variables are not expected to match between lhs and rhs;
    they are treated as sources and are mapped between each other.

    Parameters
    ----------
    lhs: tvm.relay.Expr
      One of the input Expression.

    rhs: tvm.relay.Expr
      One of the input Expression.

    Returns
    -------
    result: bool
      True iff lhs is data-flow equivalent to rhs.
    """
    return bool(_make._graph_equal(lhs, rhs))


def structural_hash(value):
    """Hash a Relay expression structurally.

    Parameters
    ----------
    expr: tvm.relay.Expr or tvm.relay.Type
      The expression to hash.

    Returns
    -------
    result: int
      The hash value
    """
    if isinstance(value, Expr):
        return int(_ir_pass._expr_hash(value))
    elif isinstance(value, Type):
        return int(_ir_pass._type_hash(value))
    else:
        msg = ("found value of type {0} expected" +
               "relay.Expr or relay.Type").format(type(value))
        raise TypeError(msg)


def fold_constant(expr):
    """Fold the constant expression in expr.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        The transformed expression.
    """
    return _ir_pass.FoldConstant(expr)


def fuse_ops(expr, opt_level=1):
    """Fuse operators in expr together.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    opt_level : int
        The level of fuse optimization.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        Transformed expression, containing fused result.
    """
    return _ir_pass.FuseOps(expr, opt_level)


def combine_parallel_conv2d(expr):
    """Fold multiple conv2d into one.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        Transformed expression
    """
    return _ir_pass.CombineParallelConv2D(expr)


def alter_op_layout(expr):
    """Alternate the layouts of operators or replace primitive operators with
    other expressions.
    This pass can be used for computing convolution in custom layouts or
    other general weight pre-transformation.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    transformed_expr : tvm.relay.Expr
        Transformed expression with alternated layout.
    """
    return _ir_pass.AlterOpLayout(expr)
