from __future__ import absolute_import as _abs
from . import expr as _expr
from . import expr_util as _expr_util
from . import domain as _dom


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
        self.rdom = None

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
        inputs = []
        if self.expr:
            def collect(e):
                if isinstance(e, _expr.TensorReadExpr):
                    inputs.append(e.tensor)
            _expr_util.visit(self.expr, collect)
        self.inputs = set(inputs)
        return self.inputs

    def infer_input_domains(self, out_domain, inputs, red_domain=None):
        """Infer the input domains of each domain in given inputs list.

        Parameters
        ----------
        out_domain : list of Range
            Domain of each dimension.

        red_domain : list of Range
            Domain of reduction variables, if this tensor
            this can only be specified if
            self.expr finishes with an ReduceExpr, and we can schedule
            over the last reduction that creates this tensor.

        Returns
        -------
        in_domains: dict Tensor->Domain
        """
        assert self.expr
        assert len(out_domain) == len(self.dim_index)
        index_domains = {
            self.dim_index[i] : out_domain[i] for i in range(len(out_domain))
        }

        begin_expr = self.expr
        if red_domain:
            if not isinstance(self.expr, _expr.ReduceExpr):
                raise ValueError("red_domain must work with tensor that stores a reduction")
            rdom = self.expr.rdom
            begin_expr = self.expr.src
            assert len(red_domain) == len(rdom.index)
            for i in range(len(red_domain)):
                index_domains[rdom.index[i]] = red_domain[i]

        iset = {}
        for t in inputs:
            assert t in self.input_tensors()
            iset[t] = []

        def prepare(e):
            if isinstance(e, _expr.ReduceExpr):
                rd = e.rdom
                for i in range(len(rd.domain)):
                    index_domains[rd.index[i]] = rd.domain[i]
            elif isinstance(e, _expr.TensorReadExpr):
                if e.tensor in iset:
                    iset[e.tensor].append(e)
        _expr_util.visit(begin_expr, prepare)
        result = {}
        for k, v in iset.items():
            dm = [None] * len(v[0].indices)
            for e in v:
                for i, idx in enumerate(e.indices):
                    dm[i] = _dom.union_range(
                        dm[i], _dom.infer_range(idx, index_domains, allow_unbind_var=False))
            result[k] = dm
        return result

    @property
    def is_rtensor(self):
        """Whether this tensor is a result of reduction.

        Returns
        -------
        is_rtensor : Whether the tensor is RTensor
        """
        return self.expr and isinstance(self.expr, _expr.ReduceExpr)
