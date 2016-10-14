from __future__ import absolute_import as _abs
from . import expr as _expr
from . import expr_util as _expr_util


class Range(object):
    """Represent a range in one dimension.
    """
    def __init__(self, begin, end=None):
        if end is None:
            end = begin
            begin = _expr.const(0)
        self.begin = _expr._symbol(begin)
        self.end = _expr._symbol(end)
        self.extent = _expr_util.simplify(end - begin)

    def __str__(self):
        return "(%s, %s)" % (
            _expr_util.format_str(self.begin),
            _expr_util.format_str(self.end))

    def __repr__(self):
        return self.__str__()

class RDom(object):
    """reduction Domain
    """
    def __init__(self, domain):
        if isinstance(domain, Range):
            domain = [domain]
        self.index = []
        self.domain = domain
        for i in range(len(domain)):
            self.index.append(_expr.Var("rd_index_%d_" % i))


"""Use list of ranges as domain"""
Domain = list
