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
"""Tensor intrinsics"""
import tvm._ffi
import tvm.tir

from tvm.runtime import Object, convert
from tvm.ir import Range
from .tensor import PlaceholderOp

from . import tensor as _tensor
from . import _ffi_api


def _get_region(tslice):
    region = []
    for idx in tslice.indices:
        if isinstance(idx, slice):
            assert idx.step is None
            region.append(Range(idx.start, idx.stop))
        else:
            if isinstance(idx, tvm.tir.IterVar):
                begin = idx.var
            else:
                begin = idx
            region.append(Range.from_min_extent(begin, 1))
    return region


@tvm._ffi.register_object
class TensorIntrin(Object):
    """Tensor intrinsic functions for certain computation.

    See Also
    --------
    decl_tensor_intrin: Construct a TensorIntrin
    """

    def __call__(self, *args, **kwargs):
        tensors = [x.tensor for x in args if isinstance(x, _tensor.TensorSlice)]
        scalar_inputs = [x for x in args if not isinstance(x, _tensor.TensorSlice)]
        regions = [_get_region(x) for x in args if isinstance(x, _tensor.TensorSlice)]
        reduce_axis = []
        if "reduce_axis" in kwargs:
            reduce_axis = kwargs["reduce_axis"]
            if not isinstance(reduce_axis, (list, tuple)):
                reduce_axis = [reduce_axis]
            reduce_axis = convert(reduce_axis)
        if scalar_inputs:
            scalar_inputs = convert(scalar_inputs)
        return _ffi_api.TensorIntrinCall(self, tensors, regions, reduce_axis, scalar_inputs)


def decl_tensor_intrin(
    op, fcompute, name="tensor_intrin", binds=None, scalar_params=None, default_buffer_params=None
):
    """Declare a tensor intrinsic function.

    Parameters
    ----------
    op: Operation
        The symbolic description of the intrinsic operation

    fcompute: lambda function of inputs, outputs-> stmt
        Specifies the IR statement to do the computation.
        See the following note for function signature of fcompute

        .. note::
             **Parameters**

             - **ins** (list of :any:`tvm.tir.Buffer`) - Placeholder for each inputs
             - **outs** (list of :any:`tvm.tir.Buffer`) - Placeholder for each outputs

             **Returns**

             - **stmt** (:any:`tvm.tir.Stmt`, or tuple of three stmts)
             - If a single stmt is returned, it represents the body
             - If tuple of three stmts are returned they corresponds to body,
               reduce_init, reduce_update

    name: str, optional
        The name of the intrinsic.

    binds: dict of :any:`Tensor` to :any:`tvm.tir.Buffer`, optional
        Dictionary that maps the Tensor to Buffer which specified the data layout
        requirement of the function. By default, a new compact buffer is created
        for each tensor in the argument.

    scalar_params: a list of variables used by op, whose values will be passed
                   as scalar_inputs when the tensor intrinsic is called.

    default_buffer_params: Optional[dict]
        Dictionary of buffer arguments to be passed when constructing a buffer.

    Returns
    -------
    intrin: TensorIntrin
        A TensorIntrin that can be used in tensorize schedule.
    """
    if not isinstance(op, _tensor.Operation):
        raise TypeError("expect Operation")
    inputs = op.input_tensors
    binds = binds if binds else {}
    tensors = list(inputs)
    for i in range(op.num_outputs):
        tensors.append(op.output(i))

    binds_list = []
    for t in inputs:
        if not isinstance(t.op, PlaceholderOp):
            raise ValueError("Do not yet support composition op")

    default_buffer_params = {} if default_buffer_params is None else default_buffer_params
    for t in tensors:
        buf = (
            binds[t]
            if t in binds
            else tvm.tir.decl_buffer(t.shape, t.dtype, t.op.name, **default_buffer_params)
        )
        binds_list.append(buf)

    if scalar_params:
        body = fcompute(binds_list[: len(inputs)], binds_list[len(inputs) :], scalar_params)
    else:
        body = fcompute(binds_list[: len(inputs)], binds_list[len(inputs) :])
        scalar_params = []
    if isinstance(body, (tvm.tir.PrimExpr, tvm.tir.Stmt)):
        body = [body]
    body = [tvm.tir.Evaluate(x) if isinstance(x, tvm.tir.PrimExpr) else x for x in body]
    if len(body) < 3:
        body += [None] * (3 - len(body))
    return _ffi_api.TensorIntrin(name, op, inputs, binds_list, scalar_params, *body)
