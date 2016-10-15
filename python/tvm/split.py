from __future__ import absolute_import as _abs
from . import expr as _expr
from . import domain as _dom
from . import tensor as _tensor


class Split(object):
    def __init__(self, dim, factor):
        self.dim = dim
        self.factor = factor
        self.loop_index = _expr.Var('loop_index_%d_' % dim)

    def infer_inner_domain(self, domain):
        if isinstance(domain, _dom.RDom):
            domain = domain.domain
        assert self.dim < len(domain)
        inner_domain = domain[:]
        dim_out_range = domain[self.dim]
        dim_inner_begin = dim_out_range.begin + self.loop_index * self.factor
        inner_domain[self.dim] = _dom.Range(dim_inner_begin, dim_inner_begin + self.factor)
        return inner_domain

