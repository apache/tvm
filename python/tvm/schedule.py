from __future__ import absolute_import as _abs
from . import domain as _dom
from . import expr as _expr
from . import expr_util as _expr_util
from . import split as _split
from . import buffer as _buffer
from . import codegen as _gen

start_point_key = '__start__'
TAB = '  '

class Schedule(object):
    """SUnit defines the compute schedule of a tensor

    Parameters
    ----------
    tensor: tensor
    """
    def __init__(self, tensor, buffer=None):
        self.tensor = tensor
        self.buffer = buffer
        self.parent = None
        #self.children = []
        self.splits = []
        self.split_attach = {start_point_key: []}
        self.implicit_splits = [_split.Split(i, 1) for i in range(tensor.ndim)]
        if isinstance(tensor.expr, _expr.ReduceExpr):
            for i in range(len(tensor.expr.rdom.domain)):
                self.implicit_splits.append(_split.Split(i, 1, rdom=True))

    def add_split(self, split):
        self.splits.append(split)
        self.split_attach[split] = []

    def set_buffer(self, buf):
        self.buffer = buf

    def attach(self, split, other):
        other.parent = self
        if split is None:
            self.split_attach[start_point_key].append(other)
        else:
            self.split_attach[split].append(other)

    def infer_inner_domain(self, domain):
        for split in self.splits:
            domain = split.infer_inner_domain(domain)
        return domain

    def realize(self, domain=None, indent=''):

        def realize_attach(lst):
            attach_tensors = [sch.tensor for sch in lst]
            attach_domains = self.tensor.infer_input_domains(domain, attach_tensors, red_domain=red_domain)
            for sch in lst:
                body.extend(sch.realize(attach_domains[sch.tensor], indent))

        # init domain and red_domain
        if domain is None:
            domain = self.tensor.domain
        red_domain = self.tensor.expr.rdom.domain if isinstance(self.tensor.expr, _expr.ReduceExpr) else None

        # init buffer shape
        if self.buffer:
            if self.buffer.scope == _buffer.Scope.Global:
                self.buffer.reshape(self.tensor.domain)
            else:
                # don't handle shared buffer for now
                self.buffer.reshape(domain)
            _buffer.BufferManager.current.bind(self.tensor, self.buffer)

        body = []

        if self.split_attach[start_point_key]:
            realize_attach(self.split_attach[start_point_key])

        # add loop conditions for splits
        for split in self.splits:
            if split.rdom:
                red_domain = split.generate_loop_condition(red_domain, body, indent)
            else:
                domain = split.generate_loop_condition(domain, body, indent)
            indent += TAB
            if self.split_attach[split]:
                realize_attach(self.split_attach[split])

        # add implicit loop conditions
        for split in self.implicit_splits:
            if split.rdom:
                red_domain = split.generate_loop_condition(red_domain, body, indent)
            else:
                domain = split.generate_loop_condition(domain, body, indent)
            indent += TAB

        # add loop body
        expr = self.tensor.expr
        global_index = [r.begin for r in domain]
        global_rdom_index = [r.begin for r in red_domain] if red_domain else []
        if expr is None:
            if self.buffer:
                lhs = self.buffer(*global_index)
                rhs = self.tensor(*global_index, flatten=True)
                body.append('%s%s = %s;' % (indent, _expr_util.format_str(lhs), _expr_util.format_str(rhs)))
        else:
            if self.buffer:
                lhs = self.buffer(*global_index)
            else:
                lhs = self.tensor(*global_index, flatten=True)

            bind_dict = {}
            for i in range(self.tensor.ndim):
                bind_dict[self.tensor.dim_index[i]] = global_index[i]
            if isinstance(expr, _expr.ReduceExpr):
                for i in range(len(expr.rdom.domain)):
                    bind_dict[expr.rdom.index[i]] = global_rdom_index[i]
            rhs = _expr_util.bind(expr, bind_dict)
            body.append('%s%s = %s;' % (indent, _expr_util.format_str(lhs), _gen.gen_code(rhs)))

        # add right brackets
        for split in self.implicit_splits:
            indent = indent[:-len(TAB)]
            body.append('%s}' % indent)
        for split in self.splits:
            indent = indent[:-len(TAB)]
            body.append('%s}' % indent)

        return body
