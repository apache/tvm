# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tensor and Operation class for computation declaration."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
from ._ffi.node import NodeBase, NodeGeneric, register_node, convert_to_node
from . import _api_internal
from . import make as _make
from . import expr as _expr


class TensorSlice(NodeGeneric, _expr.ExprOp):
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

    def asnode(self):
        """Convert slice to node."""
        return self.tensor(*self.indices)

    @property
    def dtype(self):
        """Data content of the tensor."""
        return self.tensor.dtype

@register_node
class TensorIntrinCall(NodeBase):
    """Intermediate structure for calling a tensor intrinsic."""


itervar_cls = None


@register_node
class Tensor(NodeBase, _expr.ExprOp):
    """Tensor object, to construct, see function.Tensor"""

    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError("Need to provide %d index in tensor slice" % ndim)
        indices = convert_to_node(indices)
        args = []
        for x in indices:
            if isinstance(x, _expr.Expr):
                args.append(x)
            elif isinstance(x, iter_var_cls):
                args.append(x.var)
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
            if isinstance(other, _expr.ExprOp):
                return _expr.EqualOp(self, other)
            return False
        if self.ndim == 0 and other.ndim == 0:
            raise ValueError("Equal == comparison among rank-0 tensor is ambiguous, "
                             "use Tensor.equal for content expression equvalence, "
                             "use Tensor.same_as for exact reference comparison")
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
        """The output value index the tensor corresponds to."""
        return self.__getattr__("value_index")

    @property
    def shape(self):
        """The output shape of the tensor."""
        return self.__getattr__("shape")

    @property
    def name(self):
        op = self.op
        if op.num_outputs == 1:
            return op.name
        return "%s.v%d" % (op.name, self.value_index)



class Operation(NodeBase):
    """Represent an operation that generates a tensor"""

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

    @property
    def num_outputs(self):
        """Number of outputs from this op."""
        return _api_internal._OpNumOutputs(self)

    @property
    def input_tensors(self):
        """List of input tensors to this op."""
        return _api_internal._OpInputTensors(self)


@register_node
class PlaceholderOp(Operation):
    """Placeholder operation."""


@register_node
class BaseComputeOp(Operation):
    """Compute operation."""
    @property
    def axis(self):
        """Represent the IterVar axis, defined when it is a ComputeOp"""
        return self.__getattr__("axis")

    @property
    def reduce_axis(self):
        """Represent axis of reductions, only defined when it is a ComputeOp"""
        return self.__getattr__("reduce_axis")


@register_node
class ComputeOp(BaseComputeOp):
    """Scalar operation."""
    pass


@register_node
class TensorComputeOp(BaseComputeOp):
    """Tensor operation."""


@register_node
class ScanOp(Operation):
    """Scan operation."""
    @property
    def scan_axis(self):
        """Represent the scan axis, only defined when it is a ScanOp"""
        return self.__getattr__("scan_axis")


@register_node
class ExternOp(Operation):
    """External operation."""


@register_node
class HybridOp(Operation):
    """Hybrid operation."""
    @property
    def axis(self):
        """Represent the IterVar axis, also defined when it is a HybridOp"""
        return self.__getattr__("axis")


@register_node
class Layout(NodeBase):
    """Layout is composed of upper cases, lower cases and numbers,
    where upper case indicates a primal axis and
    the corresponding lower case with factor size indicates the subordinate axis.
    For example, NCHW16c can describe a 5-D tensor of
    [batch_size, channel, height, width, channel_block].
    Here subordinate axis channel_block=16 is the factor size of the primal axis C (channel).

    Do not construct directly, use :any:`layout` instead.
    See the documentation of :any:`layout` for more details.

    See Also
    --------
    layout : Declare a layout
    """
    def __str__(self):
        return self.name

    def __repr__(self):
        return "Layout(" + self.name + ")"

    def __len__(self):
        return _api_internal._LayoutNdim(self)

    def __contains__(self, axis):
        return len(axis) == 1 and axis[0].isalpha() and axis[0] in self.name

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Layout index out of range")
        return _api_internal._LayoutGetItem(self, index)

    def index_of(self, axis):
        """Get the index of an axis

        Parameters
        ----------
        axis : str
            The axis name, need to be [a-z,A-Z]

        Returns
        -------
        index : int
            The index of the axis, -1 if not found.
        """
        return _api_internal._LayoutIndexOf(self, axis)

    def factor_of(self, axis):
        """Get the factor size of the subordinate axis.

        Parameters
        ----------
        axis : str
            The axis name, need to be [a-z,A-Z]

        Returns
        -------
        factor : int
            the size of the subordinate-axis of axis (if axis is a primal-axis),
            or the size of axis itself (if axis is a subordinate-axis).
            Return -1 if axis is not in the layout.
        """
        return _api_internal._LayoutFactorOf(self, axis)


@register_node
class BijectiveLayout(NodeBase):
    """Bijective mapping for two layouts (src-layout and dst-layout).
    It provides shape and index conversion between each other.

    Do not construct directly, use :any:`bijective_layout` instead.
    See the documentation of :any:`bijective_layout` for more details.

    See Also
    --------
    bijective_layout : Declare a bijective layout converter
    """
    def forward_index(self, index):
        """Given the indices of the src-layout, infer the dst index.

        Parameters
        ----------
        index: Array of Expr
            The indices in src-layout.

        Returns
        -------
        dst_index: Array of Expr
            The inferred indices in dst-layout.
        """
        return _api_internal._BijectiveLayoutForwardIndex(self, index)

    def backward_index(self, index):
        """Given the indices of the dst-layout, infer the src index.

        Parameters
        ----------
        index: Array of Expr
            The indices in dst-layout.

        Returns
        -------
        src_index: Array of Expr
            The inferred indices in src-layout.
        """
        return _api_internal._BijectiveLayoutBackwardIndex(self, index)

    def forward_shape(self, shape):
        """Given the shape of the src-layout, infer the dst shape.

        Parameters
        ----------
        shape: Array of Expr
            The shape in src-layout.

        Returns
        -------
        dst_shape: Array of Expr
            The inferred shape in dst-layout.
        """
        return _api_internal._BijectiveLayoutForwardShape(self, shape)

    def backward_shape(self, shape):
        """Given the shape of the dst-layout, infer the src shape.

        Parameters
        ----------
        shape: Array of Expr
            The shape in dst-layout.

        Returns
        -------
        src_shape: Array of Expr
            The inferred shape in src-layout.
        """
        return _api_internal._BijectiveLayoutBackwardShape(self, shape)
