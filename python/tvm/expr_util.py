"""Utilities to manipulate expression"""
from __future__ import absolute_import as _abs
from . import expr as _expr
from . import op as _op

def expr_with_new_children(e, children):
    """Returns same expr as e but with new children

    A shallow copy of e will happen if children differs from current children

    Parameters
    ----------
    e : Expr
       The input expression

    children : list of Expr
       The new children

    Returns
    -------
    new_e : Expr
       Expression with the new children
    """
    if children:
        if isinstance(e, _expr.BinaryOpExpr):
            return (e if children[0] == e.lhs and children[1] == e.rhs
                    else _expr.BinaryOpExpr(e.op, children[0], children[1]))
        elif isinstance(e, _expr.UnaryOpExpr):
            return e if children[0] == e.src else _expr.UnaryOpExpr(e.op, children[0])
        else:
            raise TypeError("donnot know how to handle Expr %s" % type(e))
    else:
        return e


def transform(e, f):
    """Apply f recursively to e and collect the resulr

    Parameters
    ----------
    e : Expr
       The input expression.

    f : function with signiture (e, ret_children)
       ret_children is the result of transform from children

    Returns
    -------
    result : return value of f
        The final result of transformation.
    """
    if not isinstance(e, _expr.Expr):
        raise TypeError("Cannot handle type %s" % type(e))
    return f(e , [transform(c, f) for c in e.children()])


def visit(e, f):
    """Apply f to each element of e

    Parameters
    ----------
    e : Expr
       The input expression.

    f : function with signiture (e)
    """
    assert isinstance(e, _expr.Expr)
    for c in e.children():
        visit(c, f)
    f(e)


def format_str(expr):
    """change expression to string.

    Parameters
    ----------
    expr : Expr
       Input expression

    Returns
    -------
    s : str
       The string representation of expr
    """
    def make_str(e, result_children):
        if isinstance(e, _expr.BinaryOpExpr):
            return e.op.format_str(result_children[0], result_children[1])
        elif isinstance(e, _expr.UnaryOpExpr):
            return e.op.format_str(result_children[0])
        elif isinstance(e, _expr.ConstExpr):
            return str(e.value)
        elif isinstance(e, _expr.Var):
            return e.name
        elif isinstance(e, _expr.TensorReadExpr):
            return "%s(%s)" % (e.tensor.name, ','.join(result_children))
        elif isinstance(e, _expr.ReduceExpr):
            return e.op.format_reduce_str(result_children[0], e.rdom.domain)
        else:
            raise TypeError("Do not know how to handle type " + str(type(e)))
    return transform(expr, make_str)


def simplify(expr):
    """simplify expression

    Parameters
    ----------
    expr : Expr
        Input expression

    Returns
    -------
    e : Expr
       Simplified expression
    """
    def canonical(e, result_children):
        if isinstance(e, _expr.BinaryOpExpr):
            return e.op.canonical(result_children[0], result_children[1])
        elif isinstance(e, _expr.UnaryOpExpr):
            return e.op.canonical(result_children[0])
        elif isinstance(e, _expr.ConstExpr):
            return {_op.constant_canonical_key: e.value}
        elif isinstance(e, _expr.Var):
            return {e: 1}
        else:
            raise TypeError("Do not know how to handle type " + str(type(e)))
    return _op.canonical_to_expr(transform(expr, canonical))


def bind(expr, update_dict):
    """Replace the variable in e by specification from kwarg

    Parameters
    ----------
    expr : Expr
       Input expression

    update_dict : dict of Var->Expr
       The variables to be replaced.

    Examples
    --------
    eout = bind(e, update_dict={v1: (x+1)} )
    """
    def replace(e, result_children):
        if isinstance(e, _expr.Var) and e in update_dict:
            return update_dict[e]
        else:
            return expr_with_new_children(e, result_children)
    return transform(expr, replace)
