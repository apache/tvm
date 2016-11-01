from __future__ import absolute_import as _abs
from ._ctypes._api import NodeBase, register_node
from . import _function_internal
from . import make as _make
from . import expr as _expr

@register_node
class Tensor(NodeBase):
    """Tensor object, to construct, see function.Tensor"""
    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError("Need to provide %d index in tensor slice" % ndim)
        return _make.Call(self.dtype, self.name, indices, _expr.Call.Halide, self, 0)

    @property
    def ndim(self):
        return len(self.shape)
