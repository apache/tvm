from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node, convert
from . import collections as _collections
from . import make as _make
from . import expr as _expr

@register_node
class Tensor(NodeBase):
    """Tensor object, to construct, see function.Tensor"""
    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError("Need to provide %d index in tensor slice" % ndim)
        indices = convert(indices)
        args = []

        for x in indices:
            if isinstance(x, _collections.IterVar):
                args.append(x.var)
            elif isinstance(x, _expr.Expr):
                args.append(x)
            else:
                raise ValueError("The indices must be expression")

        return _make.Call(self.dtype, self.name, args, _expr.Call.Halide, self, 0)

    @property
    def ndim(self):
        return len(self.shape)
