from __future__ import absolute_import as _abs
from . import expr as _expr
from . import expr_util as _expr_util
from . import op as _op

class Range(object):
    """Represent a range in one dimension.
    """
    def __init__(self, begin, end=None):
        if end is None:
            end = begin
            begin = _expr.const(0)
        begin = _expr_util.simplify(_expr._symbol(begin))
        end = _expr_util.simplify(_expr._symbol(end))
        self.begin = begin
        self.end = end
        self.extent = _expr_util.simplify(end - begin)

    def is_value(self):
        return isinstance(self.extent, _expr.ConstExpr) and self.extend.value == 1

    def __str__(self):
        return "(%s, %s)" % (
            _expr_util.format_str(self.begin),
            _expr_util.format_str(self.end))

    def __repr__(self):
        return self.__str__()


class RangeInferError(ValueError):
    pass


class RDom(object):
    """Reduction Domain."""
    def __init__(self, domain):
        if isinstance(domain, Range):
            domain = [domain]
        self.index = []
        self.domain = domain
        for i in range(len(domain)):
            self.index.append(_expr.Var("rd_index_%d_" % i))


"""Use list of ranges as domain"""
Domain = list


def _combine_range_binary_op(op, lhs, rhs):
    if op == _op.add:
        return Range(lhs.begin + rhs.begin, lhs.end + rhs.end - 1)
    elif op == _op.sub:
        return Range(lhs.begin - rhs.end + 1, lhs.end - rhs.begin)
    elif op == _op.mul:
        v = None
        if lhs.is_value():
            v = lhs.begin.value
            e = rhs
        elif rhs.is_value():
            v = rhs.begin.value
            e = lhs
        if v == -1:
            return Range(-e.end, -e.begin)
    raise InferRangeError("donot know how to infer range for %s" % type(op))


def infer_range(e, range_dict, allow_unbind_var=True):
    """Infer the range of result e given range of variables.

    Parameters
    ----------
    expr : Expr
       Input expression

    range_dict : dict of Var->Range
       The variables to be replaced.

    allow_unbind_var: bool
       Whether allow unbinded variables
    """
    def combine_range(e, result_children):
        if isinstance(e, _expr.ConstExpr):
            return Range(e, e + 1)
        elif isinstance(e, _expr.BinaryOpExpr):
            return _combine_range_binary_op(e.op, result_children[0], result_children[1])
        elif isinstance(e, _expr.Var):
            if e in range_dict:
                return range_dict[e]
            else:
                if allow_unbind_var:
                    return Range(e, e + 1)
                else:
                    raise ValueError("Cannot find var %s in range_dict" % e.name)
        else:
            raise InferRangeError("cannot infer range for %s" % _expr_util.format_str(e))
    return _expr_util.transform(e, combine_range)


def union_range(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    begin = _op.min(lhs.begin, rhs.begin)
    end = _op.max(rhs.end, lhs.end)
    return Range(begin, end)
