"""Tensor related abstractions"""
from __future__ import absolute_import as _abs
from ._ctypes._node import NodeBase, SliceBase, register_node, convert_to_node
from . import collections as _collections
from . import _api_internal
from . import make as _make
from . import expr as _expr

class TensorSlice(SliceBase, _expr.ExprOp):
    """Auxiliary data structure for enable slicing syntax from tensor."""
    def __init__(self, tensor, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.tensor = tensor
        self.indices = indices

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self.tensor, self.indices + indices)


@register_node
class Tensor(NodeBase):
    """Tensor object, to construct, see function.Tensor"""
    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError("Need to provide %d index in tensor slice" % ndim)
        indices = convert_to_node(indices)
        args = []
        for x in indices:
            if isinstance(x, _collections.IterVar):
                args.append(x.var)
            elif isinstance(x, _expr.Expr):
                args.append(x)
            else:
                raise ValueError("The indices must be expression")

        return _make.Call(self.dtype, self.op.name,
                          args, _expr.Call.Halide,
                          self.op, self.value_index)

    def __getitem__(self, indices):
        return TensorSlice(self, indices)

    def __hash__(self):
        return _api_internal._TensorHash(self)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return _api_internal._TensorEqual(self, other)

    @property
    def ndim(self):
        """Dimension of the tensor."""
        return len(self.shape)

    @property
    def axis(self):
        """Axis of the tensor."""
        return self.__getattr__("axis")

    @property
    def op(self):
        """The corressponding :any:`Operation`."""
        return self.__getattr__("op")

    @property
    def value_index(self):
        """The output value index the tensor corressponds to."""
        return self.__getattr__("value_index")

    @property
    def shape(self):
        """The output shape of the tensor."""
        return self.__getattr__("shape")


class Operation(NodeBase):
    """Represent an operation that generate a tensor"""
    def output(self, index):
        """Get the index-th output of the operation

        Parameters
        ----------
        index : int
            The index size.

        Returns
        -------
        out : Tensor
            The i-th output.
        """
        return _api_internal._OpGetOutput(self, index)

@register_node
class PlaceholderOp(Operation):
    """Placeholder operation."""
    pass

@register_node
class ComputeOp(Operation):
    """Compute operation."""
    pass

@register_node
class ScanOp(Operation):
    """Scan operation."""
    pass

@register_node
class ExternOp(Operation):
    """Extern operation."""
    pass
