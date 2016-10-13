from __future__ import absolute_import as _abs
from . import expr as _expr

class TensorReadExpr(_expr.Expr):
    def __init__(self, tensor, indices):
        self.tensor = tensor
        self.indices = indices

    def children(self):
        return self.indices


class Tensor(object):
    def __init__(self, ndim, fcompute=None, name=None):
        self.ndim = ndim
        if fcompute:
            arg_names = fcompute.func_code.co_varnames
            assert(len(arg_names) == ndim)
            self.dim_index = [_expr.Var(n) for n in arg_names]
            self.expr = fcompute(*self.dim_index)
        else:
            self.expr = None
            self.dim_index = None
            shape_name = '_shape'
            if name: shape_name = name + shape_name
            self.shape = tuple(_expr.Var("%s_%d_" % (shape_name, i)) for i in range(ndim))

        self.name = name if name else "TensorObj"

    def __call__(self, *indices):
        if len(indices) != self.ndim:
            raise ValueError("Need to provide %d index in tensor slice" % self.ndim)
        return TensorReadExpr(self, indices)
