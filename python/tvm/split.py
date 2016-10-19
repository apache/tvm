from __future__ import absolute_import as _abs
from . import expr as _expr
from . import expr_util as _expr_util
from . import domain as _dom
from . import tensor as _tensor


class Split(object):
    def __init__(self, dim, factor, name=None, rdom=False):
        self.dim = dim
        self.factor = factor
        self.rdom = rdom
        if name is None:
            name = 'loop_index_%d_' % dim
        self.loop_index = _expr.Var(name)

    def infer_inner_domain(self, out_domain):
        assert self.dim < len(out_domain)
        inner_domain = out_domain[:]
        dim_out_range = out_domain[self.dim]
        dim_inner_begin = dim_out_range.begin + self.loop_index * self.factor
        inner_domain[self.dim] = _dom.Range(dim_inner_begin, dim_inner_begin + self.factor)
        return inner_domain

    def generate_loop_condition(self, out_domain, body, indent):
        assert self.dim < len(out_domain)
        loop_range = _dom.Range(out_domain[self.dim].extent / self.factor)
        stmt = '%sfor (int %s = 0; %s < %s; %s += 1) {' % (
            indent,
            self.loop_index.name,
            self.loop_index.name,
            _expr_util.format_str(loop_range.end),
            self.loop_index.name)
        body.append(stmt)
        return self.infer_inner_domain(out_domain)
