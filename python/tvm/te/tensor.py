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
"""Tensor class for computation declaration."""

# pylint: disable=invalid-name
import tvm_ffi

from tvm.runtime import Object, ObjectConvertible
from tvm.tirx import DataProducer
from tvm.tirx import expr as _expr

from . import _ffi_api, _te_tensor_overload


class TensorSlice(ObjectConvertible):
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

    def asobject(self):
        """Convert slice to object."""
        return self.tensor.__call__(*self.indices)

    @property
    def dtype(self):
        """Data content of the tensor."""
        return self.tensor.dtype

    def expr_ty(self):
        """Compile-time element type of the tensor."""
        return self.tensor.expr_ty()

    def _scalar_op(self, name, *args):
        args = tuple(arg.asobject() if isinstance(arg, TensorSlice) else arg for arg in args)
        return getattr(self.asobject(), name)(*args)

    def _binary_op(self, name, other):
        result = getattr(_te_tensor_overload, name)(self, other)
        if result is NotImplemented:
            return self._scalar_op(name, other)
        return result

    def __add__(self, other):
        return self._binary_op("__add__", other)

    def __radd__(self, other):
        return self._binary_op("__radd__", other)

    def __sub__(self, other):
        return self._binary_op("__sub__", other)

    def __rsub__(self, other):
        return self._binary_op("__rsub__", other)

    def __mul__(self, other):
        return self._binary_op("__mul__", other)

    def __rmul__(self, other):
        return self._binary_op("__rmul__", other)

    def __div__(self, other):
        return self._binary_op("__div__", other)

    def __rdiv__(self, other):
        return self._binary_op("__rdiv__", other)

    def __truediv__(self, other):
        return self._binary_op("__truediv__", other)

    def __rtruediv__(self, other):
        return self._binary_op("__rtruediv__", other)

    def __floordiv__(self, other):
        return self._scalar_op("__floordiv__", other)

    def __rfloordiv__(self, other):
        return self._scalar_op("__rfloordiv__", other)

    def __mod__(self, other):
        return self._scalar_op("__mod__", other)

    def __rmod__(self, other):
        return self._scalar_op("__rmod__", other)

    def __neg__(self):
        return self._scalar_op("__neg__")

    def __lshift__(self, other):
        return self._scalar_op("__lshift__", other)

    def __rlshift__(self, other):
        return self._scalar_op("__rlshift__", other)

    def __rshift__(self, other):
        return self._scalar_op("__rshift__", other)

    def __rrshift__(self, other):
        return self._scalar_op("__rrshift__", other)

    def __and__(self, other):
        return self._scalar_op("__and__", other)

    def __rand__(self, other):
        return self._scalar_op("__rand__", other)

    def __or__(self, other):
        return self._scalar_op("__or__", other)

    def __ror__(self, other):
        return self._scalar_op("__ror__", other)

    def __xor__(self, other):
        return self._scalar_op("__xor__", other)

    def __rxor__(self, other):
        return self._scalar_op("__rxor__", other)

    def __invert__(self):
        return self._scalar_op("__invert__")

    def __lt__(self, other):
        return self._scalar_op("__lt__", other)

    def __le__(self, other):
        return self._scalar_op("__le__", other)

    def __eq__(self, other):
        return self._scalar_op("__eq__", other)

    def __ne__(self, other):
        return self._scalar_op("__ne__", other)

    def __gt__(self, other):
        return self._scalar_op("__gt__", other)

    def __ge__(self, other):
        return self._scalar_op("__ge__", other)

    def __nonzero__(self):
        return self._scalar_op("__nonzero__")

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other, span=None):
        return self._scalar_op("equal", other, span)

    def astype(self, dtype, span=None):
        result = _te_tensor_overload.astype(self, dtype, span)
        if result is NotImplemented:
            return self._scalar_op("astype", dtype, span)
        return result


@tvm_ffi.register_object("te.Tensor")
class Tensor(DataProducer, _expr.ExprOp):
    """Tensor object, to construct, see function.Tensor"""

    def __call__(self, *indices):
        ndim = self.ndim
        if len(indices) != ndim:
            raise ValueError(
                f"Need to provide {ndim} index in tensor but {len(indices)} was provided"
            )
        return _expr.ProducerLoad(self, indices)

    def __getitem__(self, indices):
        return TensorSlice(self, indices)

    def __add__(self, other):
        return _te_tensor_overload.__add__(self, other)

    def __radd__(self, other):
        return _te_tensor_overload.__radd__(self, other)

    def __sub__(self, other):
        return _te_tensor_overload.__sub__(self, other)

    def __rsub__(self, other):
        return _te_tensor_overload.__rsub__(self, other)

    def __mul__(self, other):
        return _te_tensor_overload.__mul__(self, other)

    def __rmul__(self, other):
        return _te_tensor_overload.__rmul__(self, other)

    def __div__(self, other):
        return _te_tensor_overload.__div__(self, other)

    def __rdiv__(self, other):
        return _te_tensor_overload.__rdiv__(self, other)

    def __truediv__(self, other):
        return _te_tensor_overload.__truediv__(self, other)

    def __rtruediv__(self, other):
        return _te_tensor_overload.__rtruediv__(self, other)

    def astype(self, dtype, span=None):
        result = _te_tensor_overload.astype(self, dtype, span)
        if result is NotImplemented:
            raise TypeError("TE Tensor overload astype is not registered")
        return result

    def __hash__(self):
        return _ffi_api.TensorHash(self)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            if isinstance(other, _expr.ExprOp):
                return _expr.EqualOp(self, other)
            return False
        if self.ndim == 0 and other.ndim == 0:
            raise ValueError(
                "Equal == comparison among rank-0 tensor is ambiguous, "
                "use Tensor.equal for content expression equvalence, "
                "use Tensor.same_as for exact reference comparison"
            )
        return _ffi_api.TensorEqual(self, other)

    @property
    def ndim(self):
        """Dimension of the tensor."""
        return len(self.shape)

    @property
    def dtype(self):
        """Data content of the tensor."""
        return _ffi_api.TensorDType(self)

    def expr_ty(self):
        """Compile-time element type of the tensor."""
        return self.dtype

    @property
    def name(self):
        op = self.op
        if op.num_outputs == 1:
            return op.name
        return f"{op.name}.v{self.value_index}"


@tvm_ffi.register_object("te.Operation")
class Operation(Object):
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
        return _ffi_api.OpGetOutput(self, index)

    @property
    def num_outputs(self):
        """Number of outputs from this op."""
        return _ffi_api.OpNumOutputs(self)

    @property
    def input_tensors(self):
        """List of input tensors to this op."""
        return _ffi_api.OpInputTensors(self)


@tvm_ffi.register_object("te.PlaceholderOp")
class PlaceholderOp(Operation):
    """Placeholder operation."""


@tvm_ffi.register_object("te.BaseComputeOp")
class BaseComputeOp(Operation):
    """Compute operation."""


@tvm_ffi.register_object("te.ComputeOp")
class ComputeOp(BaseComputeOp):
    """Scalar operation."""


@tvm_ffi.register_object("te.ScanOp")
class ScanOp(Operation):
    """Scan operation."""


@tvm_ffi.register_object("te.ExternOp")
class ExternOp(Operation):
    """External operation."""
