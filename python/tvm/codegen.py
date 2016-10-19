from __future__ import absolute_import as _abs
from . import buffer as _buffer
from . import expr as _expr
from . import expr_util as _expr_util

def gen_code(expr):
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
        elif isinstance(e, _expr.TensorRefExpr):
            buf = _buffer.BufferManager.current.get(e.tensor)
            if buf:
                return _expr_util.format_str(buf(*e.indices))
            return _expr_util.format_str(e.tensor(*e.indices, flatten=True))
        elif isinstance(e, _expr.ReduceExpr):
            return e.op.format_reduce_stmt_str(result_children[0])
        else:
            raise TypeError("Do not know how to handle type " + str(type(e)))
    return _expr_util.transform(expr, make_str)

