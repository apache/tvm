from __future__ import absolute_import as _abs
from . import expr as _expr
from . import expr_util as _expr_util


class Tensor(object):
    def __init__(self, ndim, fcompute=None, name=None, shape=None):
        self.ndim = ndim
        if fcompute:
            arg_names = fcompute.func_code.co_varnames
            assert(len(arg_names) == ndim)
            self.dim_index = [_expr.Var(n) for n in arg_names]
            self.expr = fcompute(*self.dim_index)
            if shape is None:
                raise ValueError("argument shape need to be given for intermediate tensor")
            self.shape = shape
        else:
            self.expr = None
            self.dim_index = None
            shape_name = '_shape'
            if name: shape_name = name + shape_name
            self.shape = shape if shape else tuple(
                _expr.Var("%s_%d_" % (shape_name, i)) for i in range(ndim))

        self.name = name if name else "TensorObj"
        self.inputs = None

    def __call__(self, *indices):
        if len(indices) != self.ndim:
            raise ValueError("Need to provide %d index in tensor slice" % self.ndim)
        return _expr.TensorReadExpr(self, indices)

    def input_tensors(self):
        """List of input tensors to this tensor.

        Returns
        -------
        inputs : list of input tensors
        """
        if self.inputs is not None:
            return self.inputs
        self.inputs = []
        if self.expr:
            def collect(e):
                if isinstance(e, _expr.TensorReadExpr):
                    self.inputs.append(e.tensor)
            _expr_util.visit(self.expr, collect)
        return self.inputs

    def infer_input_domains(self, out_domain):
        """Infer the input domains of each domain given output domains

        Parameters
        ----------
        out_domain : list of Range
            Domain of each dimension.

        Returns
        -------
        in_domains: dict Tensor->Domain
        """
        assert self.expr
        assert len(out_domain) == len(self.dim_index)
        index_domains = {
            self.dim_index[i] : out_domain[i] for i in range(len(out_domain))
        }
        def collect(e):
            if isinstance(e, _expr.TensorReadExpr):
                self.inputs.append(e.tensor)
        _expr_util.visit(self.expr, collect)
