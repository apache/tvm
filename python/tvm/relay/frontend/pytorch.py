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
# pylint: disable=import-self, too-many-lines, len-as-condition, no-else-return, unused-variable, too-many-nested-blocks
# pylint: disable=consider-iterating-dictionary, invalid-name, unused-argument, unused-variable, broad-except
# pylint: disable=import-outside-toplevel, simplifiable-if-expression, unnecessary-comprehension
"""PT: PyTorch frontend."""
import itertools
import logging
import sys

import numpy as np

import tvm
from tvm.topi.util import get_const_tuple

from .. import analysis as _analysis
from .. import expr as _expr
from .. import op as _op
from ..ty import TupleType, TensorType, Any
from ..loops import while_loop
from .. import transform
from .common import AttrCvt, get_relay_op
from .common import infer_shape as _infer_shape
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated
from .common import infer_type as _infer_type
from ..prelude import Prelude, StaticTensorArrayOps

from . import qnn_torch

__all__ = ["from_pytorch"]


# List ADT utilities
def _infer_type_with_prelude(val, prelude):
    body = _infer_type(val, prelude.mod)
    return body.checked_type


def _convert_to_list_adt(py_lst, prelude):
    elem_tys = [_infer_type_with_prelude(elem, prelude) for elem in py_lst]
    msg = "List elements should have identical types"
    assert all(map(lambda ty: ty == elem_tys[0], elem_tys)), msg

    adt_lst = prelude.nil()
    for elem in reversed(py_lst):
        adt_lst = prelude.cons(elem, adt_lst)
    return adt_lst


def _map_tensor_array_constructor(adt_lst, prelude, shape):
    static_tensor_array_ops = StaticTensorArrayOps(prelude, "float32", shape)
    static_tensor_array_ops.register()
    tensor_create = prelude.get_var_static("tensor_constructor", "float32", shape)
    return prelude.map(tensor_create, adt_lst)


def _convert_to_tensor_array(adt_lst, prelude):
    if prelude.length(adt_lst) == 0:
        return prelude.nil()

    checked_type = _infer_type_with_prelude(prelude.hd(adt_lst), prelude)
    shape = checked_type.shape
    tensor_array = _map_tensor_array_constructor(adt_lst, prelude, shape)
    return tensor_array, tuple(shape)


def _should_construct_dynamic_list(list_construct_node):
    # if this list is element-accessed or modified at runtime, generate List ADT
    def inplace_add_to_add(op_name):
        if op_name == "aten::add_":
            return "aten::add"
        else:
            return op_name

    uses = _get_uses(list_construct_node)

    for loop_use in filter(lambda use: use.user.kind() == "prim::Loop", uses):
        block_input_index = loop_use.offset - 1
        block = list(loop_use.user.blocks())[0]
        list_loop_var = list(block.inputs())[block_input_index]
        uses += _get_uses(list_loop_var.node())

    op_names = map(inplace_add_to_add, set(use.user.kind() for use in uses))

    list_ops = set(["aten::add", "aten::__getitem__"])
    intersect = list_ops.intersection(op_names)

    if len(intersect) > 0 and intersect != set(["aten::add"]):
        return True

    # if add op outputs list, it is dynamic so we need to construct List ADT
    for use in filter(lambda use: use.user.kind() in ["aten::add", "aten::add_"], uses):
        output_type = _get_node_type(use.user)
        if output_type == "ListType":
            return True

    return False


def _is_int_seq(seq):
    # TODO (t-vi): handle non-int constants? (like numpy.intXX)
    return len(seq) > 0 and all([isinstance(i, int) for i in seq])


def _is_quantized_tensor(data, prelude):
    # If a quantized Torch module is saved and loaded back, dtype will be dropped
    # Since dtypes from Torch tensors are not reliable in such cases, we use
    # Relay's type inference result to decide if an input tensor is quantized
    ty = _infer_type_with_prelude(data, prelude)
    return ty.dtype == "uint8"


# operator implementation
def _elemwise(name):
    def _impl(inputs, input_types):
        data0, data1 = _pytorch_promote_types(inputs[:2], input_types[:2])
        return get_relay_op(name)(data0, data1)

    return _impl


def _min_max_common(name_elemwise, name_reduce):
    def _impl(inputs, input_types):
        if len(inputs) == 1:
            data = _pytorch_promote_types(inputs[:1], input_types[:1])
            return get_relay_op(name_reduce)(data[0])
        elif len(inputs) >= 2 and isinstance(inputs[1], int):
            data = _pytorch_promote_types(inputs[:1], input_types[:1])
            dim = inputs[1]
            keepdims = inputs[2] if len(inputs) > 2 else False
            # also return dummy indices
            return get_relay_op(name_reduce)(data[0], axis=dim, keepdims=keepdims), None
        else:
            data0, data1 = _pytorch_promote_types(inputs[:2], input_types[:2])
            return get_relay_op(name_elemwise)(data0, data1)

    return _impl


def _max():
    return _min_max_common("maximum", "max")


def _min():
    return _min_max_common("minimum", "min")


def _unary(name):
    def _impl(inputs, input_types):
        input_type = input_types[0]
        # this is just to ensure tensor input
        (data,) = _pytorch_promote_types(inputs[:1], input_types[:1])
        return get_relay_op(name)(data)

    return _impl


def _log1p():
    def _impl(inputs, input_types):
        # 1_plus_log x = log(x + 1)
        (dtype,) = input_types
        one = _expr.const(1, dtype=dtype)
        return _op.log(inputs[0] + one)

    return _impl


def _arange():
    def _impl(inputs, input_types):
        def _get_value(val, dtype):
            # dtype is a tvm dtype
            if isinstance(val, _expr.Expr):
                try:
                    ret = _infer_value(_op.cast(val, dtype), {}).asnumpy()
                    ret = _expr.const(ret, dtype)
                except Exception:
                    ret = _op.cast(val, dtype)
            else:
                ret = _create_typed_const(val, dtype)
            return ret

        def _get_type(val, inp_type):
            if isinstance(val, _expr.Expr):
                dtype = str(_infer_type(val).checked_type)
                return dtype
            return inp_type

        # PyTorch arange uses the following type semantics:
        # - if a dtype is given, start, stop, step are converted to that dtype
        # - if no dtype is given and all args are integral, dtype is int64
        # - if no dtype is given and there is a float arg, dtype is float32
        if len(inputs) == 5:
            dtype0 = _get_type(inputs[0], input_types[0])
            if inputs[1] is not None:
                dtype = _convert_dtype_value(inputs[1])
            elif dtype0.startswith("float"):
                dtype = "float32"
            else:
                dtype = "int64"
            start = _expr.const(0, dtype)
            stop = _get_value(inputs[0], dtype)
            step = _expr.const(1, dtype)
        elif len(inputs) == 7:
            types = [_get_type(inputs[i], input_types[i]) for i in range(3)]
            if inputs[3] is not None:
                dtype = _convert_dtype_value(inputs[3])
            elif any([t.startswith("float") for t in types]):
                dtype = "float32"
            else:
                dtype = "int64"
            start = _get_value(inputs[0], dtype)
            stop = _get_value(inputs[1], dtype)
            step = _get_value(inputs[2], dtype)
        else:
            msg = "Unknown number of arguments (%d) to parse." % (len(inputs))
            raise AssertionError(msg)

        return _op.transform.arange(start=start, stop=stop, step=step, dtype=dtype)

    return _impl


def _squeeze():
    def _impl(inputs, input_types):
        data = inputs[0]
        if len(inputs) == 1:
            axis = None
        else:
            # TODO (t-vi): why is the cast to int needed? similarly elsewhere
            axis = [int(inputs[1])]

        return _op.transform.squeeze(data, axis)

    return _impl


def _unsqueeze():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        return _op.transform.expand_dims(data, int(axis), 1)

    return _impl


def _concatenate(prelude):
    def tensor_array_concat(lst, axis):
        assert axis == 0, "Tensor array concat supported only for axis 0"
        tensor_array, shape = _convert_to_tensor_array(lst, prelude)
        concat_shape = (Any(),) + shape[1:]
        concat = prelude.get_var_static("tensor_array_concat", "float32", shape)
        concatenated = concat(tensor_array)

        static_tensor_array_ops = StaticTensorArrayOps(prelude, "float32", concat_shape)
        static_tensor_array_ops.register()
        get_tensor = prelude.get_var_static("tensor_get_data", "float32", concat_shape)
        return get_tensor(concatenated)

    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        if not isinstance(data, list):
            return tensor_array_concat(data, axis)

        if isinstance(data, _expr.Expr):
            data = [data]

        return _op.tensor.concatenate(data, int(axis))

    return _impl


def _slice():
    def _impl(inputs, input_types):
        axis_dtype = "int64"
        index_size_limit = 2 ** 63 - 1
        data = inputs[0]
        dshape = _infer_shape(data)
        ndim = len(dshape)
        end = []
        for dim in dshape:
            if isinstance(dim, tvm.tir.Any):
                end = _op.shape_of(data)
                break
            end.append(int(dim))

        begin = [0] * ndim
        dim = int(inputs[1])
        stride = int(inputs[4])
        if isinstance(inputs[2], _expr.Call):
            try:
                begin[dim] = np.asscalar(_infer_value(inputs[2], {}).asnumpy().astype(np.int))
            except Exception:
                begin[dim] = inputs[2]
        else:
            begin[dim] = int(inputs[2])

        # Process begin
        if not isinstance(begin[dim], int):
            tmp = []
            for b in begin:
                if isinstance(b, int):
                    tmp.append(_op.expand_dims(_expr.const(b, axis_dtype), axis=0))
                else:
                    tmp.append(_op.cast(_op.expand_dims(b, axis=0), axis_dtype))
            begin = _op.concatenate(tmp, axis=0)
            btype = _infer_type(begin).checked_type.dtype
            if str(btype) != axis_dtype:
                begin = _op.cast(begin, axis_dtype)

        if isinstance(inputs[3], str) and inputs[3].isdigit():
            target_end = int(inputs[3])
        else:
            if isinstance(inputs[3], _expr.Expr):
                try:
                    target_end = np.asscalar(_infer_value(inputs[3], {}).asnumpy().astype(np.int))
                except Exception:
                    target_end = inputs[3]
            else:
                target_end = inputs[3]

            if isinstance(target_end, int) and target_end >= index_size_limit:
                # Quick path for original data.
                if (
                    isinstance(begin, _expr.Constant)
                    and begin.data.asnumpy().tolist()[dim] == 0
                    and stride == 1
                ):
                    return data
                target_end = dshape[dim]

        # Process end
        if isinstance(target_end, int):
            if isinstance(end, list):
                end[dim] = target_end
            else:
                all_static = True
                for i, shape_dim in enumerate(dshape):
                    if i != dim and isinstance(shape_dim, tvm.tir.Any):
                        all_static = False

                if all_static:
                    end = list(get_const_tuple(dshape))
                    end[dim] = target_end
                else:
                    target_end = _expr.const(target_end)
                    end = _op.scatter(
                        end,
                        _op.expand_dims(_expr.const(dim), axis=0),
                        _op.expand_dims(target_end, axis=0),
                        axis=0,
                    )
        else:
            end = _op.cast(_op.shape_of(data), axis_dtype)
            if not isinstance(target_end, tvm.tir.Any):
                ttype = _infer_type(target_end).checked_type.dtype
                if str(ttype) != axis_dtype:
                    target_end = _op.cast(target_end, axis_dtype)
                end = _op.scatter(
                    end,
                    _op.expand_dims(_expr.const(dim), axis=0),
                    _op.expand_dims(target_end, axis=0),
                    axis=0,
                )

        if not isinstance(end, list):
            etype = _infer_type(end).checked_type.dtype
            if str(etype) != axis_dtype:
                end = _op.cast(end, axis_dtype)

        strides = [1] * ndim
        strides[dim] = int(inputs[4])

        return _op.transform.strided_slice(
            data, begin=begin, end=end, strides=strides, slice_mode="end"
        )

    return _impl


def _split():
    def _impl(inputs, input_types):
        data = inputs[0]
        split_size = int(inputs[1])
        dim = int(inputs[2])

        split_index = split_size
        indices = []
        while split_index < _infer_shape(data)[dim]:
            indices.append(split_index)
            split_index += split_size

        return _op.split(data, indices, dim)

    return _impl


def _split_with_sizes():
    def _impl(inputs, input_types):
        data = inputs[0]
        dim = int(inputs[2])

        split_index = 0
        indices = []
        sections = inputs[1]
        for i in range(len(sections) - 1):
            split_index += sections[i]
            indices.append(split_index)

        return _op.split(data, indices, dim)

    return _impl


def _select():
    def _impl(inputs, input_types):
        data = inputs[0]
        dim = int(inputs[1])
        index = _wrap_const(inputs[2])
        return _op.transform.take(data, index, axis=dim)

    return _impl


def _take():
    def _impl(inputs, input_types):
        data = inputs[0]
        indices = _op.cast(inputs[1], "int32")

        return _op.transform.take(data, indices=indices)

    return _impl


def _topk():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = int(inputs[2])
        is_ascend = not bool(inputs[3])
        sort = bool(inputs[4])

        if isinstance(inputs[1], _expr.Expr):
            try:
                k = _infer_value(inputs[1], {}).asnumpy().tolist()
            except Exception:
                k = inputs[1]
        else:
            k = inputs[1]

        if not sort:
            msg = "Currently supports only sorted output for topk operator."
            raise AssertionError(msg)

        outs = _op.topk(data, k=k, axis=axis, is_ascend=is_ascend, ret_type="both", dtype="int64")

        return outs[0], outs[1]

    return _impl


def _reciprocal():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _expr.const(1.0, dtype=input_types[0]) / data

    return _impl


def _repeat():
    def _impl(inputs, input_types):
        data = inputs[0]
        reps = inputs[1]
        return _op.transform.tile(data, reps=reps)

    return _impl


def _repeat_interleave():
    def _impl(inputs, input_types):
        data = inputs[0]
        if isinstance(inputs[1], int):
            repeats = inputs[1]
            axis = inputs[2]
        else:
            msg = "Only repeat with one value as repeat is currently supported."
            raise AssertionError(msg)
        if axis is None:  # Flatten the data if no axis is given from torch
            data = _op.transform.reshape(data, [-1])
            axis = 0
        return _op.transform.repeat(data, repeats=repeats, axis=axis)

    return _impl


def _addcdiv():
    def _impl(inputs, input_types):
        data, t1, t2, c = _pytorch_promote_types(inputs[:4], input_types[:4])
        return data + (c * (t1 / t2))

    return _impl


def _addcmul():
    def _impl(inputs, input_types):
        data, t1, t2, c = _pytorch_promote_types(inputs[:4], input_types[:4])
        return data + (c * (t1 * t2))

    return _impl


def _where():
    def _impl(inputs, input_types):
        cond = inputs[0]
        x, y = _pytorch_promote_types(inputs[1:3], input_types[1:3])
        return _op.where(cond, x, y)

    return _impl


def _full_impl(data, fill_value, dtype):
    size = []
    need_reshape = False
    new_shape = []
    for dim in data:
        if isinstance(dim, _expr.Expr):
            if isinstance(dim, _expr.Constant):
                dim = int(dim.data.asnumpy())
                if isinstance(size, list):
                    size.append(dim)
                new_shape.append(dim)
            else:
                try:
                    dim = int(_infer_value(dim, {}).asnumpy())
                    if isinstance(size, list):
                        size.append(dim)
                    new_shape.append(dim)
                except Exception:
                    size = None
                    need_reshape = True
                    new_shape.append(0)
        else:
            if isinstance(size, list):
                size.append(dim)
            new_shape.append(dim)

    if size is None:
        tmp = []
        for dim in data:
            tmp.append(_op.cast(_op.expand_dims(dim, axis=0), "int64"))
        size = _op.concatenate(tmp, axis=0)

    out = _op.full(_expr.const(fill_value), size, dtype=dtype)
    if need_reshape:
        out = _op.reshape(out, new_shape)
    return out


def _ones(default_dtype):
    def _impl(inputs, input_types):
        data = inputs[0]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in ones op" % (type(data))
            raise AssertionError(msg)

        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = default_dtype
        return _full_impl(data, 1, dtype)

    return _impl


def _ones_like(default_dtype):
    def _impl(inputs, input_types):
        data = inputs[0]
        out = _op.ones_like(data)

        # If the input and the output datatype is different, do a cast
        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = default_dtype
        if input_types[0] != dtype:
            out = _op.cast(out, dtype)

        return out

    return _impl


def _zeros(default_dtype):
    def _impl(inputs, input_types):
        data = inputs[0]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in zeros op" % (type(data))
            raise AssertionError(msg)

        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = default_dtype
        return _full_impl(data, 0, dtype)

    return _impl


def _zeros_like(default_dtype):
    def _impl(inputs, input_types):
        data = inputs[0]
        out = _op.zeros_like(data)

        # If the input and the output datatype is different, do a cast
        if inputs[1] is not None:
            dtype = _convert_dtype_value(inputs[1])
        else:
            dtype = default_dtype
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out

    return _impl


def _full(default_dtype):
    def _impl(inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]

        import torch

        if not isinstance(data, (_expr.Expr, list, torch.Tensor, np.ndarray)):
            msg = "Data type %s could not be parsed in full op" % (type(data))
            raise AssertionError(msg)

        if inputs[2] is not None:  # dtype given
            dtype = _convert_dtype_value(inputs[2])
        else:
            # if dtype is None, torch uses a global default set by torch.set_default_tensor_type()
            dtype = default_dtype

        return _full_impl(data, fill_value, dtype)

    return _impl


def _full_like(default_dtype):
    def _impl(inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]

        out = _op.full_like(data, _expr.const(fill_value))

        # If the input and the output datatype is different, do a cast
        if inputs[2] is not None:  # dtype given
            dtype = _convert_dtype_value(inputs[2])
        else:
            # if dtype is None, torch uses a global default set by torch.set_default_tensor_type()
            dtype = default_dtype
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out

    return _impl


def _linspace():
    def _impl(inputs, input_types):
        start = inputs[0]
        stop = inputs[1]
        step = inputs[2]

        # Find the spacing between values as step
        if step != 1:
            step = (stop - start) / (step - 1)
            stop = stop + step
        else:
            stop = start + step

        dtype = "float32" if inputs[3] is not None else _convert_dtype_value(inputs[3])
        start = _create_typed_const(start, dtype)
        stop = _create_typed_const(stop, dtype)
        step = _create_typed_const(step, dtype)

        return _op.transform.arange(start=start, stop=stop, step=step, dtype=dtype)

    return _impl


def _relu(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]
        if _is_quantized_tensor(data, prelude):
            assert len(inputs) == 3, "Input quant param not found in op inputs"
            input_zero_point = _expr.const(inputs[2], dtype="int32")
            return qnn_torch.quantized_relu(data, input_zero_point)
        return _op.nn.relu(data)

    return _impl


def _prelu():
    def _impl(inputs, input_types):
        data = inputs[0]
        alpha = inputs[1]
        return _op.nn.prelu(data, alpha)

    return _impl


def _leaky_relu():
    def _impl(inputs, input_types):
        data = inputs[0]
        alpha = float(inputs[1])
        return _op.nn.leaky_relu(data, alpha)

    return _impl


def _elu():
    def _impl(inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        alpha = _expr.const(float(inputs[1]), dtype=dtype)
        return alpha * _op.nn.relu(_expr.const(1, dtype=dtype) - _op.exp(data)) + _op.nn.relu(data)

    return _impl


def _celu():
    def _impl(inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        alpha = _expr.const(float(inputs[1]), dtype=dtype)
        return alpha * _op.nn.relu(
            _expr.const(1, dtype=dtype) - _op.exp(data / alpha)
        ) + _op.nn.relu(data)

    return _impl


def _gelu():
    def _impl(inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        # gelu is data  * normcdf(data)
        # normcdf expressed as erf because we don't currently have that intrinsic
        # note that there is also a fastgelu variant approximating normcdf
        # with tanh and third order polynomials, but this is "true" gelu
        return data * (
            _expr.const(0.5, dtype=dtype)
            + _op.erf(data * _expr.const(0.5 ** 0.5, dtype=dtype)) * _expr.const(0.5, dtype=dtype)
        )

    return _impl


def _selu():
    def _impl(inputs, input_types):
        data = inputs[0]
        # https://pytorch.org/docs/stable/nn.html#selu
        dtype = input_types[0]
        alpha = _expr.const(-1.6732632423543772848170429916717, dtype=dtype)
        gamma = _expr.const(1.0507009873554804934193349852946, dtype=dtype)
        return gamma * (
            alpha * _op.nn.relu(_expr.const(1.0, dtype=dtype) - _op.exp(data)) + _op.nn.relu(data)
        )

    return _impl


def _log_sigmoid():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.log(_op.tensor.sigmoid(data))

    return _impl


def _adaptive_avg_pool_2d(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = inputs[1]

        def func(x):
            return _op.nn.adaptive_avg_pool2d(x, output_size=output_size)

        if _is_quantized_tensor(data, prelude):
            return qnn_torch.apply_with_upcast(data, func)

        return func(data)

    return _impl


def _adaptive_max_pool_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = inputs[1]

        # returns dummy indices too
        return _op.nn.adaptive_max_pool2d(data, output_size=output_size), None

    return _impl


def _adaptive_max_pool_3d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = inputs[1]
        # returns dummy indices too
        return _op.nn.adaptive_max_pool3d(data, output_size=output_size), None

    return _impl


def _adaptive_avg_pool_3d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = inputs[1]
        return _op.nn.adaptive_avg_pool3d(data, output_size=output_size)

    return _impl


def _maxpool_2d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])

        if dilation != [1, 1]:
            msg = "MaxPool2d with dilation %s is not implemented" % (str(dilation))
            raise NotImplementedError(msg)

        return _op.nn.max_pool2d(data, pool_size, strides, padding, "NCHW", ceil_mode)

    return _impl


def _maxpool_2d_with_indices():
    def _impl(inputs, input_types):
        # returns dummy indices too
        return _maxpool_2d()(inputs, input_types), None

    return _impl


def _maxpool_1d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])

        if dilation != [1]:
            msg = "MaxPool1d with dilation %s is not implemented" % (str(dilation))
            raise NotImplementedError(msg)

        return _op.nn.max_pool1d(data, pool_size, strides, padding, "NCW", ceil_mode)

    return _impl


def _maxpool_3d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        dilation = inputs[4]
        ceil_mode = int(inputs[5])
        if dilation != [1, 1, 1]:
            msg = "MaxPool3d with dilation %s is not implemented" % (str(dilation))
            raise NotImplementedError(msg)

        return _op.nn.max_pool3d(
            data, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode
        )

    return _impl


def _hardtanh():
    def _impl(inputs, input_types):
        a = inputs[0]
        tanh_min = float(inputs[1])
        tanh_max = float(inputs[2])
        return _op.tensor.clip(a, tanh_min, tanh_max)

    return _impl


def _convolution():
    def _impl(inputs, input_types):
        # Use transpose or normal
        use_transpose = True if inputs[6] == 1 else False

        data = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        strides = tuple(inputs[3])
        padding = tuple(inputs[4])
        dilation = tuple(inputs[5])

        if isinstance(weight, _expr.Expr):
            inferred_shape = _infer_shape(weight)
            weight_shape = []
            for infer in inferred_shape:
                weight_shape.append(infer)
        else:
            msg = "Data type %s could not be parsed in conv op" % (type(weight))
            raise AssertionError(msg)

        # Transposed convolutions have IOHW layout.
        if use_transpose:
            weight_shape[0], weight_shape[1] = weight_shape[1], weight_shape[0]

        channels = weight_shape[0]
        groups = int(inputs[8])

        # Check if this is depth wise convolution
        # We need to reshape weight so that Relay could recognize this is depth wise
        # weight_shape[1] is always in_channels // groups
        # For depthwise, in_channels == groups, so weight_shape[1] == 1
        # If groups > 1 but weight_shape[1] != 1, this is group convolution
        if groups > 1 and weight_shape[1] == 1:
            channel_multiplier = channels // groups
            new_weight_shape = (groups, channel_multiplier) + tuple(weight_shape[2:])
            weight = _op.transform.reshape(weight, new_weight_shape)

        kernel_size = weight_shape[2:]
        use_bias = isinstance(bias, _expr.Expr)

        if len(kernel_size) == 1:
            strides = (1,) + strides
            padding = (0,) + padding
            dilation = (1,) + dilation

        if use_transpose:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d_transpose
            else:
                conv_op = _op.nn.conv2d_transpose
        else:
            if len(kernel_size) == 3:
                conv_op = _op.nn.conv3d
            else:
                conv_op = _op.nn.conv2d

        if len(kernel_size) == 3:
            data_layout = "NCDHW"
            kernel_layout = "OIDHW"
        else:
            data_layout = "NCHW"
            kernel_layout = "OIHW"

        if len(kernel_size) == 1:
            data = _op.expand_dims(data, axis=2)
            weight = _op.expand_dims(weight, axis=2)

        conv_out = conv_op(
            data,
            weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            channels=channels,
            kernel_size=[1] + kernel_size if len(kernel_size) == 1 else kernel_size,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_layout="",
            out_dtype="",
        )
        if use_bias:
            res = _op.nn.bias_add(conv_out, bias)
        else:
            res = conv_out
        if len(kernel_size) == 1:
            res = _op.squeeze(res, axis=[2])
        return res

    return _impl


def _softmax():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        if isinstance(axis, str):
            axis = int(axis)

        return _op.nn.softmax(data, axis=axis)

    return _impl


def _threshold():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.nn.relu(data)

    return _impl


def _contiguous():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)

    return _impl


def _batch_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]

        channels = _infer_shape(data)

        if isinstance(inputs[1], _expr.Expr) and isinstance(inputs[2], _expr.Expr):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
            gamma = weight
        else:
            scale = center = False

        if not scale:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        if not center:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        moving_mean = inputs[3]
        moving_var = inputs[4]
        epsilon = float(inputs[7])

        return _op.nn.batch_norm(
            data,
            gamma,
            beta,
            moving_mean,
            moving_var,
            axis=1,
            epsilon=epsilon,
            center=center,
            scale=scale,
        )[0]

    return _impl


def _instance_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]
        channels = _infer_shape(data)

        if isinstance(inputs[1], _expr.Expr) and isinstance(inputs[2], _expr.Expr):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
            gamma = weight
        else:
            scale = center = False

        if not scale:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        if not center:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        epsilon = float(inputs[7])
        return _op.nn.instance_norm(
            data, gamma, beta, axis=1, epsilon=epsilon, center=center, scale=scale
        )

    return _impl


def _get_dims(data):
    import torch

    if isinstance(data, _expr.Expr):
        dims = _infer_shape(data)
    elif isinstance(data, list):
        dims = data
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        dims = data.shape
    else:
        msg = "Data type %s could not be parsed" % type(data)
        raise AssertionError(msg)
    return dims


def _layer_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        ndims = len(_get_dims(inputs[1]))
        assert ndims == 1, "Support only normalization over last one dimension."

        return _op.nn.layer_norm(
            data,
            gamma=inputs[2],
            beta=inputs[3],
            axis=-1,
            epsilon=float(inputs[4]),
            center=True,
            scale=True,
        )

    return _impl


def _group_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        gamma = inputs[2]
        beta = inputs[3]
        num_groups = inputs[1]
        epsilon = float(inputs[4])

        return _op.nn.group_norm(
            data,
            gamma=gamma,
            beta=beta,
            num_groups=num_groups,
            axis=1,
            epsilon=epsilon,
            center=True,
            scale=True,
        )

    return _impl


def _transpose(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]

        import torch

        if isinstance(data, _expr.Expr):
            ndims = len(_infer_shape(data, prelude.mod))
        elif isinstance(data, list):
            ndims = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            ndims = data.shape
        else:
            msg = "Data type %s could not be parsed in transpose op" % (type(data))
            raise AssertionError(msg)

        if isinstance(data, tvm.runtime.NDArray):
            ndims = len(data.shape)
        axes = list(range(ndims))

        num_inputs = len(inputs)

        if num_inputs == 1:
            if ndims >= 2:
                axes[-1] = ndims - 2
                axes[-2] = ndims - 1
            if not isinstance(data, _expr.Expr):
                data = _expr.const(data)

        elif num_inputs == 3:
            parse = lambda i: ndims * (i < 0) + i
            src, dst = [parse(int(inputs[i])) for i in [1, 2]]
            axes[src] = dst
            axes[dst] = src
        else:
            axes = inputs[1]
        return _op.transform.transpose(data, axes)

    return _impl


def _flatten():
    def _impl(inputs, input_types):
        data = inputs[0]
        start = int(inputs[1])
        end = int(inputs[2])
        dshape = get_const_tuple(_infer_shape(data))
        ndim = len(dshape)
        if end < 0:
            end += ndim
        new_shape = [0] * start

        new_shape.append(-1)
        squeeze_axes = []
        for i in range(start + 1, end + 1):
            new_shape.append(1)
            squeeze_axes.append(i)
        for _ in range(end + 1, ndim):
            new_shape.append(0)
        out = _op.reshape(data, new_shape)
        if squeeze_axes:
            out = _op.squeeze(out, axis=squeeze_axes)
        return out

    return _impl


def _addmm():
    def _impl(inputs, input_types):
        input_mat = inputs[0]
        mat1 = inputs[1]
        data_type = input_types[1]
        mat2 = inputs[2]

        beta = inputs[3]
        alpha = inputs[4]

        if not isinstance(alpha, _expr.Expr) and alpha != 1:
            alpha = _create_typed_const(alpha, data_type)
            mat1 *= alpha

        if not isinstance(beta, _expr.Expr) and beta != 1:
            beta = _create_typed_const(beta, data_type)
            mat2 *= beta

        transposed_mat2 = _op.transform.transpose(mat2, axes=[1, 0])

        units = _infer_shape(transposed_mat2)[0]
        dense_out = _op.nn.dense(mat1, transposed_mat2, units=units)

        return dense_out + input_mat

    return _impl


def _size(prelude):
    def _impl_dynamic(inp, axis):
        shape_dynamic = _op.shape_of(inp, dtype="int32")
        if axis is not None:
            return _op.take(shape_dynamic, _expr.const(axis), 0)
        return shape_dynamic

    def _impl(inputs, input_types):
        shape = _infer_shape(inputs[0], prelude.mod)
        axis = None
        if len(inputs) > 1:
            axis = int(inputs[1])

        if any(map(lambda s: isinstance(s, tvm.tir.expr.Any), shape)):
            if axis is None or isinstance(shape[axis], tvm.tir.expr.Any):
                return _impl_dynamic(inputs[0], axis)

        if axis is not None:
            return _expr.const(shape[axis])
        return _expr.const(shape)

    return _impl


def _numtotensor():
    def _impl(inputs, input_types):
        val = inputs[0]
        dtype = input_types[0]

        if isinstance(val, _expr.Expr):
            return val

        if isinstance(val, tvm.tir.IntImm):
            val = val.__int__()
            dtype = int

        arr = val * np.ones([]).astype(dtype)
        return arr

    return _impl


def _tensortonum():
    def _impl(inputs, input_types):
        return inputs[0]

    return _impl


def _view():
    def _impl(inputs, input_types):
        data = inputs[0]

        if len(inputs) == 3:
            shape_inp = [inputs[1], _infer_shape(inputs[2])[0]]
        else:
            if isinstance(inputs[1], list):
                shape_inp = inputs[1]
            else:
                shape_inp = _infer_shape(inputs[1])
        new_shape = shape_inp
        for i, shape in enumerate(shape_inp):
            if isinstance(shape, _expr.Expr):
                val = _infer_value_simulated(shape, {})
                new_shape[i] = np.asscalar(val.asnumpy())

        return _op.transform.reshape(data, new_shape)

    return _impl


def _reshape():
    def _impl(inputs, input_types):
        data = inputs[0]
        new_shape = inputs[1]

        tmp_shape = []
        is_dyn = False
        for s in new_shape:
            if isinstance(s, _expr.Constant):
                tmp_shape.append(int(s.data.asnumpy()))
            elif isinstance(s, _expr.Expr):
                try:
                    dim = int(_infer_value(s, {}).asnumpy())
                    tmp_shape.append(dim)
                except Exception:
                    is_dyn = True
                    tmp_shape.append(s)
            else:
                tmp_shape.append(s)

        if is_dyn:
            new_shape = []
            for i, s in enumerate(tmp_shape):
                if not isinstance(s, _expr.Expr):
                    s = _expr.const(s, "int64")
                else:
                    s = _op.cast(s, "int64")
                new_shape.append(_op.expand_dims(s, axis=0))
            new_shape = _op.concatenate(new_shape, axis=0)
        else:
            new_shape = tmp_shape
        return _op.transform.reshape(data, new_shape)

    return _impl


def _pixel_shuffle(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]
        upscale_factor = inputs[1]
        upscale_squared = upscale_factor * upscale_factor
        b, c, h, w = _infer_shape(data)
        assert (
            c % upscale_squared == 0
        ), "input channel should be divisible by square of upscale_factor"

        ndims = len(_infer_shape(data, prelude.mod))
        axes = list(range(ndims))
        num_inputs = len(inputs)
        oc = c // upscale_squared
        oh = h * upscale_factor
        ow = w * upscale_factor

        new_shape = [b, oc, upscale_factor, upscale_factor, h, w]
        out_shape = [b, oc, oh, ow]

        data = _op.transform.reshape(data, new_shape)
        # The data will be transposed to
        # [b, oc, h, upscale_factor, w, upscale_factor]
        # for further reshape
        axes = [0, 1, 4, 2, 5, 3]
        data = _op.transform.transpose(data, axes)
        return _op.transform.reshape(data, out_shape)

    return _impl


def _clone():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)

    return _impl


def _log_softmax():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        return _op.nn.log_softmax(data, axis)

    return _impl


def _sigmoid():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.sigmoid(data)

    return _impl


def _softplus():
    def _impl(inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        beta = _expr.const(float(inputs[1]), dtype=dtype)
        return _op.log(_op.exp(inputs[0] * beta) + _expr.const(1.0, dtype=dtype)) / beta

    return _impl


def _avg_pool2d(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        ceil_mode = int(inputs[4])
        count_include_pad = int(inputs[5])

        def func(x):
            return _op.nn.avg_pool2d(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )

        if _is_quantized_tensor(data, prelude):
            return qnn_torch.apply_with_upcast(data, func)

        return func(data)

    return _impl


def _avg_pool3d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = inputs[1]
        strides = inputs[2] if inputs[2] else pool_size
        padding = inputs[3]
        ceil_mode = int(inputs[4])
        count_include_pad = int(inputs[5])

        return _op.nn.avg_pool3d(
            data,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    return _impl


def _dropout():
    def _impl(inputs, input_types):
        data = inputs[0]
        rate = float(inputs[1])

        return _op.nn.dropout(data, rate)

    return _impl


def _reduce(name):
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = None
        keepdims = False

        if len(inputs) > 2:  # default, torch have only data, axis=None, keepdims=False
            if isinstance(inputs[1], int):
                axis = int(inputs[1])
            elif _is_int_seq(inputs[1]):
                axis = inputs[1]
            else:
                axis = list(_infer_shape(inputs[1]))
            keepdims = bool(inputs[2])

        return get_relay_op(name)(data, axis=axis, keepdims=keepdims)

    return _impl


def _norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        dtype = input_types[0]
        axis = None
        keepdims = False
        if len(inputs) > 3:
            axis = inputs[2]
            keepdims = bool(inputs[3])

        order = inputs[1]
        if order == np.inf:
            return _op.reduce.max(_op.abs(data), axis=axis, keepdims=keepdims)
        elif order == np.NINF:
            return _op.reduce.min(_op.abs(data), axis=axis, keepdims=keepdims)
        else:
            reci_order = _expr.const(1.0 / order, dtype=dtype)
            order = _expr.const(order)
            return _op.power(
                _op.reduce.sum(_op.power(_op.abs(data), order), axis=axis, keepdims=keepdims),
                reci_order,
            )

    return _impl


def _frobenius_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = None
        keepdims = False
        if len(inputs) > 2:
            axis = inputs[1]
            keepdims = bool(inputs[2])

        return _op.sqrt(_op.reduce.sum((data * data), axis=axis, keepdims=keepdims))

    return _impl


def _std():
    def _impl(inputs, input_types):
        data = inputs[0]
        if len(inputs) == 2:
            axis = None
            keepdims = False
            unbiased = bool(inputs[1])
        else:
            axis = inputs[1]
            keepdims = bool(inputs[3])
            unbiased = bool(inputs[2])

        return _op.reduce.std(data, axis=axis, keepdims=keepdims, unbiased=unbiased)

    return _impl


def _variance():
    def _impl(inputs, input_types):
        data = inputs[0]
        if len(inputs) == 2:
            axis = None
            keepdims = False
            unbiased = bool(inputs[1])
        else:
            axis = inputs[1]
            keepdims = bool(inputs[3])
            unbiased = bool(inputs[2])

        return _op.reduce.variance(data, axis=axis, keepdims=keepdims, unbiased=unbiased)

    return _impl


def _mean(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]

        if inputs[1]:
            axis = inputs[1]
        else:
            axis = None

        if len(inputs) > 2 and inputs[2]:
            keepdims = int(inputs[2])
        else:
            keepdims = False
        if len(inputs) > 3 and inputs[3]:
            exclude = int(inputs[3])
        else:
            exclude = False

        def func(x):
            return _op.mean(x, axis, keepdims, exclude)

        if _is_quantized_tensor(data, prelude):
            assert len(inputs) == 6, "Input quant param not found in op inputs"
            input_scale = _expr.const(inputs[4])
            input_zero_point = _expr.const(inputs[5])
            return qnn_torch.quantized_mean(data, input_scale, input_zero_point, func)

        return func(data)

    return _impl


def _chunk(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]

        num_chunks = int(inputs[1])
        axis = int(inputs[2])

        if isinstance(data, _expr.Expr):
            inferred_shape = _infer_shape(data, prelude.mod)

        shape = []
        for infer in inferred_shape:
            shape.append(infer)

        dim = int(shape[axis])

        if dim % num_chunks:
            unif_size = int(dim / (num_chunks - 1))
        else:
            unif_size = int(dim / num_chunks)

        chunks = []
        for i in range(0, dim, unif_size):
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = i
            end[axis] = i + unif_size
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin=begin, end=end, strides=stride)
            chunks.append(chunk_out)

        if dim % num_chunks:
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = unif_size * (num_chunks - 1)
            end[axis] = dim
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin=begin, end=end, strides=stride)
            chunks.append(chunk_out)

        return chunks

    return _impl


def _matmul(prelude):
    def _impl(inputs, input_types):

        inputs_0 = inputs[0]
        inputs_1 = inputs[1]

        # Need to check input shape as batch matmul must be supported.
        a_shape = _infer_shape(inputs_0, prelude.mod)
        b_shape = _infer_shape(inputs_1, prelude.mod)

        # When performing a batch matmul, we need to properly handle N-dim shapes.
        if len(a_shape) > 2 or len(b_shape) > 2:
            # Convert a and b into 3 dimensional tensors.
            a = _op.reshape(inputs_0, [-1, a_shape[-2], a_shape[-1]])
            b = _op.reshape(inputs_1, [-1, b_shape[-2], b_shape[-1]])
            # Broadcast b to match batch size of a
            new_b_shape = list(_infer_shape(b, prelude.mod))
            new_a_shape = _infer_shape(a, prelude.mod)
            if new_a_shape[0] > new_b_shape[0]:
                new_b_shape[0] = new_a_shape[0]
                b = _op.broadcast_to(b, new_b_shape)
            # Transpose matrix dimensions of b.
            b = _op.transpose(b, [0, 2, 1])
            # Perform a batch matmul.
            output = _op.nn.batch_matmul(a, b)
            # Reshape output to original dimensions.
            return _op.reshape(output, [*a_shape[:-2], a_shape[-2], b_shape[-1]])

        # Otherwise a simple dense op will get the job done.
        if len(b_shape) == 1:
            input_1 = _op.expand_dims(inputs_1, 0, 1)
        else:
            input_1 = _op.transpose(inputs_1, axes=(1, 0))

        out = _op.nn.dense(inputs_0, input_1)

        if len(b_shape) == 1:
            out = _op.squeeze(out, axis=[-1])

        return out

    return _impl


def _expand():
    def _impl(inputs, input_types):
        data_in = inputs[0]
        shape = list(_infer_shape(data_in))

        ndims = len(shape)
        sizes = inputs[1]
        out = data_in

        out_dims = len(sizes)
        if ndims < out_dims:
            num_newaxis = out_dims - ndims
            out = _op.expand_dims(out, axis=0, num_newaxis=num_newaxis)
            shape = [1] * num_newaxis + shape

        for i in range(out_dims):
            if sizes[i] != -1 and shape[i] == 1:
                if not isinstance(sizes[i], int):
                    sizes[i] = int(_infer_value(sizes[i], {}).asnumpy())
                out = _op.repeat(out, sizes[i], axis=i)

        return out

    return _impl


def _int():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], _expr.Expr):
            return inputs[0]
        return int(inputs[0])

    return _impl


def _identity():
    def _impl(inputs, input_types):
        return inputs[0]

    return _impl


def _none():
    def _impl(inputs, input_types):
        return None

    return _impl


def _pad(mode):
    def _impl(inputs, input_types):
        data = inputs[0]
        if isinstance(inputs[1], list):
            pad_list = inputs[1]
        else:
            pad_list = list(_infer_shape(inputs[1]))

        # initialize paddings based on input len
        pad_len = len(_infer_shape(data)) * 2
        paddings = [0] * pad_len

        if len(pad_list) >= 2:
            paddings[-1] = pad_list[1]
            paddings[-2] = pad_list[0]
        if len(pad_list) >= 4:
            paddings[-3] = pad_list[3]
            paddings[-4] = pad_list[2]
        if len(pad_list) >= 6:
            paddings[-5] = pad_list[5]
            paddings[-6] = pad_list[4]

        # group into tuple of 2 ints
        paddings = [paddings[i : i + 2] for i in range(0, len(paddings), 2)]

        const_paddings = []
        for pad in paddings:
            const_paddings.append([])
            for p in pad:
                if not isinstance(p, int):
                    p = int(_infer_value(p, {}).asnumpy())
                const_paddings[-1].append(p)

        if mode == "constant":
            return _op.nn.pad(data, const_paddings, pad_value=inputs[2], pad_mode=mode)
        else:
            return _op.nn.pad(data, const_paddings, pad_mode=mode)

    return _impl


def _clamp():
    def _impl(inputs, input_types):
        data = inputs[0]
        amin = inputs[1] if inputs[1] else np.finfo(np.float32).min
        amax = inputs[2] if inputs[2] else np.finfo(np.float32).max
        return _op.clip(data, amin, amax)

    return _impl


def _to():
    def _impl(inputs, input_types):
        data = inputs[0]
        dtype = inputs[1] if inputs[1] is not None and not isinstance(inputs[1], str) else inputs[2]
        # special handling for aten::to(data, 6, _, _, _) case
        # 6 means dtype = float
        # this happens when converting upsampling with scale factor
        cast_map = {
            6: "float32",
            7: "float64",
            3: "int32",
            4: "int64",
        }

        cast_func = {6: float, 7: float, 3: int, 4: int}

        ret = data
        if isinstance(data, _expr.Expr):
            actual_dtype = str(_infer_type(data).checked_type.dtype)
            if dtype in cast_map and cast_map[dtype] != actual_dtype:
                ret = _op.cast(data, cast_map[dtype])
        elif dtype in cast_map:
            ret = cast_func[dtype](data)

        return ret

    return _impl


def _upsample(method, prelude):
    def _impl(inputs, input_types):
        out_size = []
        for size in inputs[1]:
            if not isinstance(size, int):
                out_size.append(int(_infer_value(size, {}).asnumpy()))
            else:
                out_size.append(size)

        data = inputs[0]

        if len(inputs) > 2:
            align_corners = inputs[2]
        else:
            align_corners = False

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        def func(x):
            return _op.image.resize(x, out_size, "NCHW", method, coord_trans)

        if _is_quantized_tensor(data, prelude):
            import torch
            from packaging import version

            # Torch version > 1.4 changed upsampling API
            if version.parse(torch.__version__) > version.parse("1.4.0"):
                num_inputs = 7
            else:
                num_inputs = 5

            assert len(inputs) == num_inputs, "Input quant param not found in op inputs"

            input_scale = _expr.const(inputs[-2])
            input_zero_point = _expr.const(inputs[-1])
            return qnn_torch.quantized_upsample(data, input_scale, input_zero_point, func)
        return func(data)

    return _impl


def _upsample3d(method):
    def _impl(inputs, input_types):
        if isinstance(inputs[1], _expr.Var):
            out_size = _infer_shape(inputs[1])
        elif _is_int_seq(inputs[1]):
            out_size = inputs[1]
        elif isinstance(inputs[1], list):
            infer_res = [_infer_value(size, {}) for size in inputs[1]]
            out_size = [np.asscalar(res.asnumpy().astype(np.int)) for res in infer_res]

        data = inputs[0]

        if len(inputs) > 2:
            align_corners = inputs[2]
        else:
            align_corners = False

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return _op.image.resize3d(data, out_size, "NCDHW", method, coord_trans)

    return _impl


def _expand_as():
    def _impl(inputs, input_types):
        target = inputs[1]
        t0 = _infer_type(inputs[0]).checked_type.dtype
        t1 = _infer_type(inputs[1]).checked_type.dtype
        if str(t0) != str(t1):
            target = _op.cast(target, t0)
        return _op.broadcast_to_like(inputs[0], target)

    return _impl


def _Bool():
    def _impl(inputs, input_types):
        assert len(inputs) == 1
        return inputs[0]

    return _impl


def _Float():
    def _impl(inputs, input_types):
        assert len(inputs) == 1
        return _op.cast(inputs[0], "float32")

    return _impl


def _mm():
    def _impl(inputs, input_types):
        return _op.nn.dense(inputs[0], inputs[1])

    return _impl


def _bitwise_not():
    def _impl(inputs, input_types):
        data = inputs[0]
        # The input tensor must be of integral or Boolean types.
        # For bool tensors, it computes the logical NOT
        if input_types[0] == "bool":
            out = _op.logical_not(_op.cast(data, "bool"))
        else:
            out = _op.bitwise_not(_op.cast(data, "int"))

        return out

    return _impl


def _bitwise_xor():
    def _impl(inputs, input_types):
        lhs = inputs[0]
        rhs = inputs[1]
        lhs = _op.cast(lhs, "bool") if input_types[0] == "bool" else _op.cast(lhs, "int")
        rhs = _op.cast(rhs, "bool") if input_types[1] == "bool" else _op.cast(rhs, "int")

        return _op.bitwise_xor(lhs, rhs)

    return _impl


def _logical_not():
    def _impl(inputs, input_types):
        data = inputs[0]

        return _op.logical_not(_op.cast(data, "bool"))

    return _impl


def _logical_xor():
    def _impl(inputs, input_types):
        lhs = _op.cast(inputs[0], "bool")
        rhs = _op.cast(inputs[1], "bool")

        return _op.logical_xor(lhs, rhs)

    return _impl


def _list_getitem(prelude):
    def _impl(inputs, input_types):
        return prelude.nth(inputs[0], _wrap_const(inputs[1]))

    return _impl


def _list_len(prelude):
    def _impl(inputs, input_types):
        return prelude.length(inputs[0])

    return _impl


def _type_as():
    def _impl(inputs, input_types):
        assert len(inputs) == 2
        assert len(input_types) == 2
        return _op.cast(inputs[0], input_types[1])

    return _impl


def _gather():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        indices = inputs[2]

        return _op.gather(data, axis, indices)

    return _impl


def _add(prelude):
    # add_ is overloaded for tensor add and list concat
    def _impl(inputs, input_types):
        if input_types[0] == "ListType":
            return prelude.concat(inputs[0], inputs[1])
        return _elemwise("add")(inputs, input_types)

    return _impl


def _tensor_array_stack(prelude):
    def _impl(inputs, input_types):
        dim = inputs[1]
        assert dim == 0, "stacking on a dynamic tensor list only supported on a first axis"
        tensor_array, shape = _convert_to_tensor_array(inputs[0], prelude)

        stacked_shape = (Any(),) + shape
        stack = prelude.get_var_static("tensor_array_stack", "float32", shape)
        stacked = stack(tensor_array)

        static_tensor_array_ops = StaticTensorArrayOps(prelude, "float32", stacked_shape)
        static_tensor_array_ops.register()
        get_tensor = prelude.get_var_static("tensor_get_data", "float32", stacked_shape)
        return get_tensor(stacked)

    return _impl


def _stack(prelude):
    def _impl(inputs, input_types):
        if isinstance(inputs[0], list):
            # a static python list of tensors
            dim = inputs[1]
            return _op.stack(inputs[0], dim)
        else:
            # List ADT case
            assert isinstance(inputs[0], _expr.Expr)
            ty = _infer_type_with_prelude(inputs[0], prelude)
            list_ty = prelude.mod.get_global_type_var("List")
            msg = "The input list is expected to be List ADT"
            assert isinstance(ty, tvm.ir.TypeCall) and ty.func == list_ty, msg
            return _tensor_array_stack(prelude)(inputs, input_types)

    return _impl


def _rsub():
    def _impl(inputs, input_types):
        data0, data1 = _pytorch_promote_types(inputs[:2], input_types[:2])

        # TODO (t-vi): should this also be part of the type promotion?
        alpha = _expr.const(float(inputs[2]))

        # note: rsub means data0 and data1 swap places
        return get_relay_op("subtract")(data1, alpha * data0)

    return _impl


def _embedding():
    def _impl(inputs, input_types):
        weight = inputs[0]
        indices = inputs[1]

        return _op.take(weight, indices.astype("int32"), axis=0)

    return _impl


def _one_hot():
    def _impl(inputs, input_types):
        indices = inputs[0].astype("int32")
        num_classes = inputs[1]
        if num_classes == -1:
            msg = "Inferring the number of classes is not yet supported."
            raise NotImplementedError(msg)

        dtype = "int32"
        on_value = tvm.relay.const(1.0, dtype)
        off_value = tvm.relay.const(0.0, dtype)

        return _op.one_hot(indices, on_value, off_value, num_classes, -1, dtype)

    return _impl


def _index():
    def _impl(inputs, input_types):
        data = inputs[0]
        indices = inputs[1]
        return _op.adv_index([data] + indices)

    return _impl


def _meshgrid():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.meshgrid(data, indexing="ij")

    return _impl


def _nms(prelude):
    def _impl(inputs, input_types):
        boxes = inputs[0]
        scores = inputs[1]
        iou_threshold = inputs[2]

        # Generate data with shape (1, num_anchors, 5)
        scores = AttrCvt(op_name="expand_dims", extras={"axis": -1, "num_newaxis": 1})([scores], {})

        # Prepare input data for get_valid_counts
        data = _op.concatenate([scores, boxes], -1)
        data = _op.expand_dims(data, 0, 1)
        # Leverage get_valid_counts to sort the data and clear invalid boxes
        ct, data, indices = get_relay_op("get_valid_counts")(
            data, score_threshold=-1.0, id_index=-1, score_index=0
        )

        # Perform Non-Maximum Suppression,
        # PyTorch NMS doesn't have parameter top_k and max_output_size
        score_index = 0
        top_k = max_out_size = -1
        nms_ret = get_relay_op("non_max_suppression")(
            data=data,
            valid_count=ct,
            indices=indices,
            max_output_size=max_out_size,
            iou_threshold=iou_threshold,
            force_suppress=True,
            top_k=top_k,
            coord_start=1,
            score_index=score_index,
            id_index=-1,
            return_indices=True,
            invalid_to_bottom=False,
        )

        # squeeze the two outputs of nms for strided_slice
        size = get_relay_op("squeeze")(nms_ret[1], axis=[1])
        data_slice = get_relay_op("squeeze")(nms_ret[0], axis=[0])

        # strided slice to get the dynamic result
        return get_relay_op("strided_slice")(
            data_slice, begin=_expr.const([0]), end=size, slice_mode="size"
        )

    return _impl


def _logsumexp():
    def _impl(inputs, input_types):
        data = _pytorch_promote_types(inputs[:1], input_types[:1])
        dim_list = inputs[1]
        keepdim = inputs[2] if len(inputs) > 2 else False
        # dim is output of prim::ListConstruct, even if it is int in python code
        assert isinstance(dim_list, list), "dim is expected to be a list"
        return _op.logsumexp(data[0], axis=dim_list, keepdims=keepdim)

    return _impl


def _roi_align(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]
        boxes = inputs[1]

        output_size = (inputs[3], inputs[4])
        spatial_scale = inputs[2]
        sample_ratio = inputs[5]
        aligned = False if len(inputs) < 7 else inputs[6]

        if aligned:
            boxes -= _expr.const(0.5 / spatial_scale)

        return _op.vision.roi_align(data, boxes, output_size, spatial_scale, sample_ratio)

    return _impl


def _unbind():
    def _impl(inputs, input_types):
        data = inputs[0]
        dim = int(inputs[1])
        ishapes = _infer_shape(data)
        if dim >= len(ishapes):
            msg = "Please check input dim, it shouldn't" "be greater than or equal to rank."
            raise AttributeError(msg)

        selections = ishapes[dim]
        res_split = _op.split(data, selections, dim)
        # squeeze each split piece to get same shape as aten::unbind
        # TODO (yongwww): add new op to avoid the squeeze overhead
        ret = []
        for i in range(selections):
            ret.append(_op.transform.squeeze(res_split[i], axis=[dim]))
        ret = _expr.TupleWrapper(_expr.Tuple(ret), selections)
        return ret

    return _impl


def _shape_as_tensor(prelude):
    def _impl(inputs, input_types):
        is_symbolic_shape = False
        input_shape = _infer_shape(inputs[0], prelude.mod)
        for axis in input_shape:
            if not isinstance(axis, (int, tvm.tir.IntImm)):
                is_symbolic_shape = True
                break

        if is_symbolic_shape:
            ret = _op.shape_of(inputs[0], dtype="int64")
        else:
            ret = _expr.const(np.array(input_shape), dtype="int64")

        return ret

    return _impl


def _logical_and():
    def _impl(inputs, input_types):
        lhs = _op.cast(inputs[0], "bool")
        rhs = _op.cast(inputs[1], "bool")

        return _op.logical_and(lhs, rhs)

    return _impl


def _nonzero(is_numpy_style):
    def _impl(inputs, input_types):
        data = inputs[0]
        ret = _op.transform.argwhere(data)

        if is_numpy_style or (len(inputs) > 1 and inputs[1]):
            # TODO(kevinthesun): Support this by adding unbind op
            # ret = _unbind()([ret, 0], None)
            raise RuntimeError("as_tuple is not supported yet for nonzero.")
        return ret

    return _impl


def _scatter():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        index = inputs[2]
        src = inputs[3]
        return _op.transform.scatter(data, index, src, axis)

    return _impl


def _scalar_tensor():
    def _impl(inputs, input_types):
        data = inputs[0]
        cast_map = {
            6: "float32",
            7: "float64",
            3: "int32",
            4: "int64",
        }
        type_key = inputs[1]
        if isinstance(data, _expr.Constant):
            data = data.data.asnumpy().tolist()
        return _expr.const(data, cast_map[type_key])

    return _impl


def _interpolate():
    def _impl(inputs, input_types):
        if isinstance(inputs[1], _expr.Expr):
            out_size = inputs[1]
        elif isinstance(inputs[1], list):
            try:
                infer_res = [_infer_value(size, {}) for size in inputs[1]]
                out_size = [np.asscalar(res.asnumpy().astype(np.int)) for res in infer_res]
            except Exception:
                h = _op.expand_dims(inputs[1][0], axis=0)
                w = _op.expand_dims(inputs[1][1], axis=0)
                out_size = _op.concatenate([h, w], axis=0)

        data = inputs[0]
        align_corners = inputs[4]
        method = inputs[3]
        if method.startswith("nearest"):
            method = "nearest_neighbor"

        if method == "nearest_neighbor":
            coord_trans = "asymmetric"
        elif align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        return _op.image.resize(data, out_size, "NCHW", method, coord_trans)

    return _impl


def _pytorch_result_type(dtypes, non_tensor_inputs):
    """This promotes TVM dtypes like PyTorch would"""
    import torch

    dtype_map = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if len(dtypes) > 0:
        result_type = dtypes[0]
        for dt in dtypes[1:]:
            if dt != result_type:  # we don't want to work with same types as we
                # don't do quantized here (which cannot be promoted?)
                result_type = _convert_data_type(
                    str(
                        torch.result_type(
                            torch.zeros((), dtype=dtype_map[result_type]),
                            torch.zeros((), dtype=dtype_map[dt]),
                        )
                    )
                )
    else:
        result_type = "bool"  # this is the smallest type...
    for inp in non_tensor_inputs:
        result_type = _convert_data_type(
            str(torch.result_type(torch.zeros((), dtype=dtype_map[result_type]), inp))
        )
    return result_type


def _pytorch_promote_types(inputs, dtypes):
    """This promotes TVM inputs with TVM dtypes passed like PyTorch would"""
    actual_dtypes = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, _expr.Expr):
            idt = _infer_type(inp).checked_type.dtype
            actual_dtypes.append(idt)
        else:
            actual_dtypes.append(dtypes[i])
    dtypes = actual_dtypes
    tensor_dtypes = [dt for inp, dt in zip(inputs, dtypes) if not np.isscalar(inp)]
    non_tensor_inputs = [inp for inp in inputs if np.isscalar(inp)]
    result_type = _pytorch_result_type(tensor_dtypes, non_tensor_inputs)
    results = []
    for inp, dt in zip(inputs, dtypes):
        if np.isscalar(inp):
            results.append(_expr.const(inp, dtype=result_type))
        elif dt == result_type:
            results.append(inp)
        else:
            results.append(_op.cast(inp, result_type))
    return results


# Helper functions for operator implementation
def _convert_dtype_value(val):
    """converts a PyTorch the PyTorch numeric type id to a torch scalar type."""
    convert_torch_dtype_map = {
        7: "torch.float64",
        6: "torch.float32",
        5: "torch.float16",
        4: "torch.int64",
        3: "torch.int32",
        2: "torch.int16",
        1: "torch.int8",
        0: "torch.unit8",
        None: "torch.int64",
    }  # Default is torch.int64
    if val in convert_torch_dtype_map:
        return _convert_data_type(convert_torch_dtype_map[val])
    else:
        msg = "Torch data type value %d is not handled yet." % (val)
        raise NotImplementedError(msg)


def _convert_data_type(input_type, default_dtype=None):
    """converts the PyTorch scalar type input_type to a TVM dtype.
    optionally, default_dtype can be a TVM dtype that is used
    if input_type is None (but not when it is unknown)"""
    if input_type is None and default_dtype is not None:
        return default_dtype

    input_type = input_type.lower()
    if input_type in ["double", "torch.float64"]:
        return "float64"
    elif input_type in ["float", "torch.float32"]:
        return "float32"
    elif input_type in ["half", "torch.float16"]:
        return "float16"
    elif input_type in ["long", "torch.int64"]:
        return "int64"
    elif input_type in ["int", "torch.int32"]:
        return "int32"
    elif input_type in ["short", "torch.int16"]:
        return "int16"
    elif input_type in ["char", "torch.int8"]:
        return "int8"
    elif input_type in ["byte", "torch.uint8"]:
        return "uint8"
    elif input_type in ["quint8", "torch.quint8"]:
        return "quint8"
    elif input_type in ["qint8", "torch.qint8"]:
        return "qint8"
    elif input_type in ["qint32", "torch.qint32"]:
        return "qint32"
    elif input_type in ["bool", "torch.bool"]:
        return "bool"
    elif input_type in ["str"]:
        return "str"
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(input_type))
    return "float32"  # Never reached


def _create_typed_const(data, dtype):
    """create a (scalar) constant of given value and dtype.
    dtype should be a TVM dtype"""

    if dtype == "float64":
        typed_data = _expr.const(np.float64(data), dtype=dtype)
    elif dtype == "float32":
        typed_data = _expr.const(np.float32(data), dtype=dtype)
    elif dtype == "float16":
        typed_data = _expr.const(np.float16(data), dtype=dtype)
    elif dtype == "int64":
        typed_data = _expr.const(np.int64(data), dtype=dtype)
    elif dtype == "int32":
        typed_data = _expr.const(np.int32(data), dtype=dtype)
    elif dtype == "int16":
        typed_data = _expr.const(np.int16(data), dtype=dtype)
    elif dtype == "int8":
        typed_data = _expr.const(np.int8(data), dtype=dtype)
    elif dtype == "uint8":
        typed_data = _expr.const(np.uint8(data), dtype=dtype)
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(dtype))
    return typed_data


def _wrap_const(c):
    if not isinstance(c, (_expr.Expr, list, tvm.tir.expr.Any)):
        return _expr.const(c)
    return c


# Operator mappings
def _get_convert_map(prelude, default_dtype):
    convert_map = {
        "aten::pixel_shuffle": _pixel_shuffle(prelude),
        "aten::device": _none(),
        "prim::device": _none(),
        "aten::sub": _elemwise("subtract"),
        "aten::sub_": _elemwise("subtract"),
        "aten::max": _max(),
        "aten::min": _min(),
        "aten::mul": _elemwise("multiply"),
        "aten::mul_": _elemwise("multiply"),
        "aten::pow": _elemwise("power"),
        "aten::arange": _arange(),
        "aten::meshgrid": _meshgrid(),
        "aten::div": _elemwise("divide"),
        "aten::div_": _elemwise("divide"),
        "aten::floor_divide": _elemwise("floor_divide"),
        "aten::true_divide": _elemwise("divide"),
        "aten::addcdiv": _addcdiv(),
        "aten::addcmul": _addcmul(),
        "aten::ones": _ones(default_dtype),
        "aten::ones_like": _ones_like(default_dtype),
        "aten::zeros": _zeros(default_dtype),
        "aten::zeros_like": _zeros_like(default_dtype),
        "aten::full": _full(default_dtype),
        "aten::full_like": _full_like(default_dtype),
        "aten::linspace": _linspace(),
        "aten::reciprocal": _reciprocal(),
        "aten::repeat": _repeat(),
        "aten::repeat_interleave": _repeat_interleave(),
        "aten::to": _to(),
        "aten::squeeze": _squeeze(),
        "aten::unsqueeze": _unsqueeze(),
        "aten::cat": _concatenate(prelude),
        "aten::slice": _slice(),
        "aten::split": _split(),
        "aten::split_with_sizes": _split_with_sizes(),
        "aten::select": _select(),
        "aten::take": _take(),
        "aten::where": _where(),
        "aten::topk": _topk(),
        "aten::relu": _relu(prelude),
        "aten::relu_": _relu(prelude),
        "aten::prelu": _prelu(),
        "aten::leaky_relu": _leaky_relu(),
        "aten::leaky_relu_": _leaky_relu(),
        "aten::elu": _elu(),
        "aten::elu_": _elu(),
        "aten::celu": _celu(),
        "aten::gelu": _gelu(),
        "aten::selu": _selu(),
        "aten::log_sigmoid": _log_sigmoid(),
        "aten::adaptive_avg_pool2d": _adaptive_avg_pool_2d(prelude),
        "aten::adaptive_max_pool2d": _adaptive_max_pool_2d(),
        "aten::max_pool2d": _maxpool_2d(),
        "aten::max_pool2d_with_indices": _maxpool_2d_with_indices(),
        "aten::max_pool1d": _maxpool_1d(),
        "aten::max_pool3d": _maxpool_3d(),
        "aten::hardtanh": _hardtanh(),
        "aten::hardtanh_": _hardtanh(),
        "aten::_convolution": _convolution(),
        "aten::softmax": _softmax(),
        "aten::threshold": _threshold(),
        "aten::threshold_": _threshold(),
        "aten::contiguous": _contiguous(),
        "aten::batch_norm": _batch_norm(),
        "aten::instance_norm": _instance_norm(),
        "aten::layer_norm": _layer_norm(),
        "aten::group_norm": _group_norm(),
        "aten::transpose": _transpose(prelude),
        "aten::transpose_": _transpose(prelude),
        "aten::t": _transpose(prelude),
        "aten::flatten": _flatten(),
        "aten::addmm": _addmm(),
        "aten::size": _size(prelude),
        "aten::view": _view(),
        "aten::reshape": _reshape(),
        "aten::clone": _clone(),
        "aten::log_softmax": _log_softmax(),
        "aten::sigmoid": _sigmoid(),
        "aten::softplus": _softplus(),
        "aten::avg_pool2d": _avg_pool2d(prelude),
        "aten::avg_pool3d": _avg_pool3d(),
        "aten::dropout": _dropout(),
        "aten::dropout_": _dropout(),
        "aten::feature_dropout": _dropout(),
        "aten::alpha_dropout": _dropout(),
        "aten::mean": _mean(prelude),
        "aten::chunk": _chunk(prelude),
        "aten::matmul": _matmul(prelude),
        "aten::bmm": _matmul(prelude),
        "aten::expand": _expand(),
        "aten::Int": _int(),
        "prim::NumToTensor": _numtotensor(),
        "prim::ImplicitTensorToNum": _tensortonum(),
        "aten::ScalarImplicit": _tensortonum(),
        "aten::constant_pad_nd": _pad("constant"),
        "aten::reflection_pad1d": _pad("reflect"),
        "aten::reflection_pad2d": _pad("reflect"),
        "aten::replication_pad1d": _pad("edge"),
        "aten::replication_pad2d": _pad("edge"),
        "aten::replication_pad3d": _pad("edge"),
        "aten::permute": _transpose(prelude),
        "aten::sum": _reduce("sum"),
        "aten::prod": _reduce("prod"),
        "aten::argmin": _reduce("argmin"),
        "aten::argmax": _reduce("argmax"),
        "aten::norm": _norm(),
        "aten::frobenius_norm": _frobenius_norm(),
        "aten::std": _std(),
        "aten::var": _variance(),
        "aten::abs": _unary("abs"),
        "aten::neg": _unary("negative"),
        "aten::cos": _unary("cos"),
        "aten::cosh": _unary("cosh"),
        "aten::sin": _unary("sin"),
        "aten::sinh": _unary("sinh"),
        "aten::tan": _unary("tan"),
        "aten::tanh": _unary("tanh"),
        "aten::acos": _unary("acos"),
        "aten::asin": _unary("asin"),
        "aten::atan": _unary("atan"),
        "aten::log": _unary("log"),
        "aten::log2": _unary("log2"),
        "aten::log10": _unary("log10"),
        "aten::log1p": _log1p(),
        "aten::exp": _unary("exp"),
        "aten::erf": _unary("erf"),
        "aten::trunc": _unary("trunc"),
        "aten::sign": _unary("sign"),
        "aten::sqrt": _unary("sqrt"),
        "aten::rsqrt": _unary("rsqrt"),
        "aten::ceil": _unary("ceil"),
        "aten::floor": _unary("floor"),
        "aten::round": _unary("round"),
        "aten::isfinite": _unary("isfinite"),
        "aten::isinf": _unary("isinf"),
        "aten::isnan": _unary("isnan"),
        "aten::clamp": _clamp(),
        "aten::clamp_": _clamp(),
        "aten::detach": _identity(),
        "aten::upsample_bilinear2d": _upsample("bilinear", prelude),
        "aten::upsample_nearest2d": _upsample("nearest_neighbor", prelude),
        "aten::upsample_trilinear3d": _upsample3d("trilinear"),
        "aten::upsample_nearest3d": _upsample3d("nearest_neighbor"),
        "aten::expand_as": _expand_as(),
        "aten::lt": _elemwise("less"),
        "aten::gt": _elemwise("greater"),
        "aten::le": _elemwise("less_equal"),
        "aten::ge": _elemwise("greater_equal"),
        "aten::ne": _elemwise("not_equal"),
        "aten::eq": _elemwise("equal"),
        "aten::logical_not": _logical_not(),
        "aten::logical_xor": _logical_xor(),
        "aten::bitwise_not": _bitwise_not(),
        "aten::bitwise_xor": _bitwise_xor(),
        "aten::Bool": _Bool(),
        "aten::Float": _Float(),
        "aten::adaptive_avg_pool3d": _adaptive_avg_pool_3d(),
        "aten::adaptive_max_pool3d": _adaptive_max_pool_3d(),
        "aten::rsub": _rsub(),
        "aten::embedding": _embedding(),
        "aten::one_hot": _one_hot(),
        "aten::mm": _matmul(prelude),
        "aten::add": _add(prelude),
        "aten::add_": _add(prelude),
        "aten::stack": _stack(prelude),
        "aten::__getitem__": _list_getitem(prelude),
        "aten::len": _list_len(prelude),
        "aten::type_as": _type_as(),
        "aten::gather": _gather(),
        "aten::index_select": _select(),
        "aten::index": _index(),
        "torchvision::nms": _nms(prelude),
        "aten::logsumexp": _logsumexp(),
        "torchvision::roi_align": _roi_align(prelude),
        "aten::unbind": _unbind(),
        "aten::__and__": _logical_and(),
        "aten::_shape_as_tensor": _shape_as_tensor(prelude),
        "aten::nonzero": _nonzero(False),
        "aten::nonzero_numpy": _nonzero(True),
        "aten::scatter": _scatter(),
        "aten::scalar_tensor": _scalar_tensor(),
        "aten::__interpolate": _interpolate(),
    }
    return convert_map


def _run_jit_passes(graph):
    """ The inline pass is necessary to unwrap prim::CallMethod """
    import torch

    torch._C._jit_pass_inline(graph)


def _get_tensor_and_var(torch_tensor, name):
    tensor = tvm.nd.array(torch_tensor.cpu().numpy())
    var = _expr.var(name, shape=tensor.shape, dtype=tensor.dtype)
    return tensor, var


def _get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()


def _get_output_names(node):
    return [output.debugName() for output in node.outputs()]


def _get_input_names(node_or_graph):
    return [inp.debugName() for inp in node_or_graph.inputs()]


def _get_op_inputs(op_node, outputs):
    return [outputs[name] for name in _get_input_names(op_node)]


def _get_node_type(node):
    assert node.outputsSize() == 1
    return node.output().type().kind()


def _get_uses(node):
    uses = []
    for output in node.outputs():
        uses += output.uses()
    return uses


def _get_users(node):
    return [use.user for use in _get_uses(node)]


def _report_missing_conversion(op_names, convert_map):
    """ Check if all ops in an input graph are supported by TVM """
    known_ops = [
        "prim::Constant",
        "prim::GetAttr",
        "prim::ListConstruct",
        "prim::ListUnpack",
        "prim::TupleConstruct",
        "prim::TupleUnpack",
        "prim::If",
        "prim::Loop",
    ]
    known_ops += list(convert_map.keys())
    known_ops += list(qnn_torch.convert_map.keys())

    missing = [op_name for op_name in op_names if op_name not in known_ops]

    if missing:
        msg = "The following operators are not implemented: {}".format(missing)
        raise NotImplementedError(msg)


def _getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert len(attribute_names) == 1
    attr_name = node.s(attribute_names[0])
    return attr_name


def _getattr_full_name(getattrs):
    return ".".join([_getattr_attr_name(node) for node in getattrs])


def _get_pytorch_value_type(typ, default_dtype="float32"):
    kind = typ.kind()
    if kind == "TensorType":
        if typ.scalarType() is None:
            # Tensor's type can be unknown if we use torch.jit.script(...)
            # Defaults can be passed in, if not it is float32
            logging.warning("Untyped Tensor found, assume it is %s", default_dtype)
            return default_dtype
        else:
            return _convert_data_type(typ.scalarType())

    elif kind == "ListType":
        return "ListType"
    elif kind in ["IntType", "FloatType", "BoolType", "StringType", "OptionalType"]:
        pt_dtype = str(typ).lower()
        dtype = pt_dtype if pt_dtype == "OptionalType" else _convert_data_type(pt_dtype)
        return dtype
    else:
        return "UnsupportedType"


def _get_input_types(op_node, outputs, default_dtype="float32"):
    """Returns a TVM dtype for each input nodes derived from the torch type"""
    in_types = []
    for inp in op_node.inputs():
        if inp.node().kind() == "prim::GetAttr":
            # GetAttr nodes always return None when we call scalarType() on it
            name = inp.debugName()
            assert name in outputs
            if isinstance(outputs[name], _expr.Var):
                in_types.append(outputs[name].type_annotation.dtype)
            else:
                # For quantized modules with parameters, here we would get
                # "prim::GetAttr[name="_packed_params"]". Since the dtype corresponding to
                # _packed_params is not needed by quantized ops, we return an arbitrary type.
                in_types.append(default_dtype)
        else:
            in_types.append(_get_pytorch_value_type(inp.type(), default_dtype=default_dtype))

    return in_types


def _get_constant(node):
    """ Retrieve a constant associated with this prim::Constant node """
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)

    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()

        if ty == "IntType":
            return node.i(attr_name)
        elif ty == "BoolType":
            return bool(node.i(attr_name))
        elif ty in ["FloatType", "LongType"]:
            return node.f(attr_name)
        elif ty in ["TensorType", "CompleteTensorType"]:
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if len(tensor.shape) == 0:  # tensor(0.1)
                # TODO(t-vi): When is this needed?
                return tensor.item()
            return _wrap_const(tensor.numpy())
        elif ty in ["DeviceObjType", "StringType"]:
            return node.s(attr_name)
        elif ty == "FunctionType":
            return None
        else:
            raise NotImplementedError("Unsupported type: %s" % ty)
    else:
        assert num_attributes == 0
        return None


def _get_operator_nodes(nodes):
    """ Returns torch IR nodes that need conversion to Relay """
    ops = []
    # Traverse nodes and add to graph
    for node in nodes:
        if node.outputsSize() > 1:
            node_name = "_".join(_get_output_names(node))
        else:
            node_name = _get_output_name(node)

        if node.kind() != "prim::GetAttr":
            ops.append((node_name, node))

    return ops


def _get_relay_input_vars(graph, input_shapes, prelude, is_module=True, default_dtype="float32"):
    """
    Return Relay vars from input shapes and create entries based on
    expected graph inputs - to allow translation
    """

    graph_inputs = list(graph.inputs())
    if is_module:
        # a module has "self" as first input, which we do not need/want
        graph_inputs = graph_inputs[1:]

    if not isinstance(input_shapes, list):
        msg = "Graph inputs input_shapes should be a list"
        raise RuntimeError(msg)

    if len(graph_inputs) != len(input_shapes):
        msg = "PyTorch has {} inputs and input_shapes lists {}.".format(
            len(graph_inputs), len(input_shapes)
        )
        raise RuntimeError(msg)

    def get_relay_ty(ishape, pt_type):
        if pt_type.kind() == "TensorType":
            if not (_is_int_seq(ishape) or len(ishape) == 0):
                msg = "Shape for Tensors must be lists of ints"
                raise RuntimeError(msg)
            if (pt_type.dim() is not None and pt_type.dim() != len(ishape)) or (
                pt_type.sizes() is not None
                and any([s1 != s2 for s1, s2 in zip(pt_type.sizes(), ishape)])
            ):
                msg = "Shapes of input list and information in the graph do not match"
                raise RuntimeError(msg)
            pt_dtype = pt_type.scalarType()
            dtype = _convert_data_type(pt_dtype, default_dtype=default_dtype)
            return TensorType(ishape, dtype)
        elif pt_type.kind() == "TupleType":
            if not isinstance(ishape, tuple):
                msg = "Shapes for tuples must be tuples"
                raise RuntimeError(msg)
            return TupleType(
                [get_relay_ty(elem, pt_t) for elem, pt_t in zip(ishape, pt_type.elements())]
            )
        elif pt_type.kind() == "ListType":
            if not isinstance(ishape, list):
                msg = "Shapes for lists must be lists"
                raise RuntimeError(msg)
            pt_elemtype = pt_type.getElementType()
            elem_tys = [get_relay_ty(s, pt_elemtype) for s in ishape]
            if len(elem_tys) > 0 and not all(map(lambda ty: ty == elem_tys[0], elem_tys)):
                msg = "List elements need have identical types"
                raise RuntimeError(msg)
            return prelude.l(elem_tys[0])
        elif pt_type.kind() == "OptionalType":
            # we do not support None yet, so we fill in the type
            return get_relay_ty(ishape, pt_type.getElementType())
        # TODO: scalar inputs
        raise NotImplementedError("unsupported input type")

    input_vars = {}

    for num, inp in enumerate(input_shapes):
        if not isinstance(inp, tuple):
            msg = "Graph input {} is not a tuple".format(num)
            raise RuntimeError(msg)
        if len(inp) != 2 or not isinstance(inp[0], str):
            msg = "Graph input {} is not valid, expected ('name', shape)".format(inp)
            raise RuntimeError(msg)

    input_types = [
        (name, get_relay_ty(shape, gi.type()))
        for (name, shape), gi in zip(input_shapes, graph_inputs)
    ]

    ir_inputs = [i.debugName() for i in graph_inputs]
    for ir_input, (name, itype) in zip(ir_inputs, input_types):
        inp = _expr.var(name, type_annotation=itype)
        # Translate from graph input to user input name
        input_vars[ir_input] = inp

    return input_vars


def _unpack_tuple(tup):
    def unpack(tup, num_fields):
        return [_expr.TupleGetItem(tup, i) for i in range(num_fields)]

    if isinstance(tup, _expr.Tuple):
        return unpack(tup, len(tup.fields))
    elif isinstance(tup.type_annotation, TupleType):
        return unpack(tup, len(tup.type_annotation.fields))
    # shouldn't happen
    assert False


def _get_free_vars_from_block(block):
    block_inp_names = _get_input_names(block)
    bound_names = block_inp_names
    free_vars = set()

    for node in block.nodes():
        inp_names = _get_input_names(node)
        list_diff = [name for name in inp_names if name not in bound_names]
        free_vars.update(list_diff)
        bound_names += _get_output_names(node)

    return free_vars


def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """

    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = _get_users(current)

        if not users or terminate(users):
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """

    def terminate(users):
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]
        return len(next_attrs) == 0

    return get_use_chains(root_getattr_node, terminate)


def convert_params(graph, state_dict):
    """
    Return Relay vars and TVM NDArrays for input parameters
    A chain of prim::GetAttr nodes is processed one at a time
    """
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    params = {}
    param_tensors = {}
    packed_param_map = {}
    vars_by_name = {}
    seen = set()

    for node in getattr_nodes:
        if _get_output_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(_get_output_name, getattrs))

            full_attr = _getattr_full_name(getattrs)
            full_attr_node_name = _get_output_name(getattrs[-1])

            if full_attr.endswith("_packed_params"):  # for quantized models
                err_msg = "parameter %s not found in state dict" % full_attr
                assert full_attr in state_dict, err_msg
                packed_param_map[full_attr_node_name] = full_attr
            elif full_attr in state_dict:
                if full_attr in vars_by_name:
                    var = vars_by_name[full_attr]
                else:
                    torch_tensor = state_dict[full_attr]
                    tensor, var = _get_tensor_and_var(torch_tensor, full_attr)
                    param_tensors[full_attr] = tensor
                    vars_by_name[full_attr] = var
                params[full_attr_node_name] = var

    return params, param_tensors, packed_param_map


def convert_block(block, outputs, convert_map, prelude, default_dtype="float32"):
    """ Translate Torch "Block", used for prim::If and prim::Loop """
    ops = _get_operator_nodes(block.nodes())
    ret_names = _get_input_names(block.returnNode())
    return convert_operators(
        ops, outputs, ret_names, convert_map, prelude, default_dtype=default_dtype
    )


def convert_if(if_node, outputs, convert_map, prelude, default_dtype="float32"):
    """ Translate Torch prim::If to Relay If """
    cond = outputs[if_node.inputsAt(0).debugName()]
    blocks = list(if_node.blocks())
    true_branch = convert_block(
        blocks[0], outputs, convert_map, prelude, default_dtype=default_dtype
    )
    false_branch = convert_block(
        blocks[1], outputs, convert_map, prelude, default_dtype=default_dtype
    )
    assert len(true_branch) == 1 and len(false_branch) == 1
    return _expr.If(cond, true_branch[0], false_branch[0])


def convert_loop(loop_node, outputs, convert_map, prelude):
    """ Translate Torch prim::Loop to Relay while_loop """

    def get_input(index):
        ivalue = loop_node.inputsAt(index)
        inode = ivalue.node()
        if inode.kind() == "prim::Constant":
            return _expr.const(_get_constant(inode))
        var_name = ivalue.debugName()
        assert var_name in outputs
        return _wrap_const(outputs[var_name])

    # Refer to the spec for prim::Loop below
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
    # The first input: %max_trip_count
    # The second input: %initial_condition
    # The rest of input: loop variables
    max_loop_count = get_input(0)
    init_cond = get_input(1)
    num_loop_var = len(list(loop_node.inputs())) - 2
    init_vals = [get_input(i + 2) for i in range(num_loop_var)]

    # while loop has always max_loop_count being int64 max
    # max_loop_count.data (tvm.runtime.NDArray) is -1, so _get_constant again
    is_while_loop = (
        isinstance(max_loop_count, _expr.Constant)
        and _get_constant(loop_node.inputsAt(0).node()) == sys.maxsize
    )

    if is_while_loop:
        loop_iter_dtype = "bool"
        # while loop with non input dependent condition such as while i < 10:
        # init_cond is int, need to cast to bool to type check
        if isinstance(init_cond, _expr.Constant):
            init_cond = _op.cast(init_cond, "bool")
        init_loop_iter_val = init_cond
    else:
        loop_iter_dtype = "int32"
        # always count from 0
        init_loop_iter_val = _expr.const(0, dtype="int32")

    body_block = list(loop_node.blocks())[0]
    block_input_names = _get_input_names(body_block)
    num_block_inputs = len(block_input_names)
    name_val_pairs = list(zip(block_input_names, [init_loop_iter_val] + init_vals))
    outputs.update(name_val_pairs)

    def get_var(name, val):
        if val:
            checked_type = _infer_type_with_prelude(val, prelude)
            if hasattr(checked_type, "shape"):
                shape = get_const_tuple(checked_type.shape)
                actual_shape = []
                for dim in shape:
                    if isinstance(dim, int) and dim == 0:
                        actual_shape.append(Any())
                    else:
                        actual_shape.append(dim)
                return _expr.var(name, shape=actual_shape, dtype=checked_type.dtype)
            else:
                return _expr.var(name, type_annotation=checked_type)
        return _expr.var(name)

    loop_iter_var = _expr.var(block_input_names[0], shape=(), dtype=loop_iter_dtype)
    loop_vars = [get_var(name, val) for name, val in name_val_pairs[1:]]

    # Add non constant free variables to loop variables to prevent code blow up
    # Without this, if there are two for loops in a row, which often happens
    # if the outer loop is unrolled, the computation corresponding to the first for loop
    # is inlined inside loop body, turning O(N) + O(N) computation into O(N^2).
    # This issue was found when converting from Stacked LSTM test. Torch does not add the output
    # of the eariler loop into loop variables of the next loop.
    # So the variable corresponding to the first loop output appears free in the second loop body.
    free_vars = [
        var
        for var in _get_free_vars_from_block(body_block)
        if var in outputs
        and not isinstance(outputs[var], (_expr.Constant, int, float, str))
        and outputs[var]
    ]

    prev_outputs = {}
    for name in free_vars:
        prev_output = outputs[name]
        new_loop_var = get_var(name, prev_output)
        prev_outputs[name] = prev_output
        outputs[name] = new_loop_var
        loop_vars.append(new_loop_var)
        init_vals.append(prev_output)

    def cond(*current_vals):
        i = current_vals[0]

        if is_while_loop:
            return _op.equal(i, _expr.const(True, "bool"))

        return _op.less(i, max_loop_count)

    def body(*current_vals):
        # Update loop variables using the prev iteration outputs
        assert len(current_vals) == num_block_inputs + len(free_vars)

        for (i, val) in enumerate(current_vals):
            if i < num_block_inputs:
                outputs[block_input_names[i]] = val
            else:
                outputs[free_vars[i - num_block_inputs]] = val

        block_outputs = convert_block(body_block, outputs, convert_map, prelude)
        block_outputs += [outputs[name] for name in free_vars]

        if not is_while_loop:
            # iter var increment implicit in torch, so do it manually
            # for while loop, block_outputs[0] is already a boolean,
            # the result of termination check
            incr = _expr.const(1, dtype="int32")
            block_outputs[0] = current_vals[0] + incr

        return block_outputs

    loop = while_loop(cond, [loop_iter_var] + loop_vars, body)
    loop_val = loop(init_loop_iter_val, *init_vals)

    # restore original output values for free vars
    outputs.update(prev_outputs)

    # The first element is a loop counter or boolean condition, ignore it
    return [_expr.TupleGetItem(loop_val, i + 1) for i in range(num_loop_var)]


def convert_operators(operators, outputs, ret_names, convert_map, prelude, default_dtype="float32"):
    """ Convert each Torch IR operators to Relay equivalent """
    for node_name, op_node in operators:
        operator = op_node.kind()
        inputs = _get_op_inputs(op_node, outputs)

        if operator == "prim::Constant":
            outputs[node_name] = _get_constant(op_node)
        elif operator == "prim::ListConstruct" and _should_construct_dynamic_list(op_node):
            outputs[node_name] = _convert_to_list_adt(inputs, prelude)
        elif operator == "prim::ListConstruct":
            # This assumes that no more elements will be appended to this list
            # In this case, we keep the Python list
            outputs[node_name] = inputs
        elif operator == "prim::TupleConstruct":
            outputs[node_name] = _expr.Tuple(inputs)
        elif operator in ["prim::ListUnpack", "prim::TupleUnpack"]:
            assert len(inputs) == 1
            if isinstance(inputs[0], (list, _expr.TupleWrapper)):
                unpacked = inputs[0]
            else:
                unpacked = _unpack_tuple(inputs[0])
            outputs.update(zip(_get_output_names(op_node), unpacked))
        elif operator == "prim::If":
            if_out = convert_if(op_node, outputs, convert_map, prelude, default_dtype=default_dtype)
            outputs[node_name] = if_out
        elif operator == "prim::Loop":
            loop_out = convert_loop(op_node, outputs, convert_map, prelude)
            unpacked_names = _get_output_names(op_node)
            assert len(loop_out) == len(unpacked_names)
            outputs.update(zip(unpacked_names, loop_out))
        else:
            relay_op = convert_map[operator]
            relay_out = relay_op(
                inputs, _get_input_types(op_node, outputs, default_dtype=default_dtype)
            )

            if isinstance(relay_out, tuple):
                # This is for torch operators that return multiple outputs
                # See _adaptive_max_2d above for example
                out_names = _get_output_names(op_node)
                outputs.update(zip(out_names, relay_out))
            else:
                assert op_node.outputsSize() == 1
                outputs[node_name] = relay_out

    return [_wrap_const(outputs[ret_name]) for ret_name in ret_names]


def get_all_op_names(graph):
    """ Return all operator names in the input graph """
    nodes = list(graph.nodes())
    prim_with_blocks = ["prim::If", "prim::Loop"]
    for prim in prim_with_blocks:
        prim_nodes = graph.findAllNodes(prim, recurse=True)
        for prim_node in prim_nodes:
            for block in prim_node.blocks():
                nodes += block.nodes()
    return set(node.kind() for node in nodes)


def from_pytorch(script_module, input_shapes, custom_convert_map=None, default_dtype="float32"):
    """Load PyTorch model in the form of a scripted PyTorch model and convert into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input))

    input_shapes : List of tuples of input name and input dimensions
        Graph level input shape list
        The same input names need to be used for deployment, so choose easy to
        remember names (such as: input0, input1)

    custom_convert_map: Dictionary of str to Relay op
        A custom op conversion map in the same format as _convert_map above

    Returns
    -------
    mod : tvm.relay.Module
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime.NDArray
        Dict of converted parameters stored in tvm.runtime.ndarray format
    """
    import torch

    mod = tvm.IRModule()
    prelude = Prelude(mod)

    convert_map = _get_convert_map(prelude, default_dtype)

    graph = script_module.graph.copy()
    _run_jit_passes(graph)

    if custom_convert_map:
        convert_map.update(custom_convert_map)

    op_names = get_all_op_names(graph)
    _report_missing_conversion(op_names, convert_map)

    is_module = isinstance(script_module, torch.jit.ScriptModule)
    params = script_module.state_dict() if is_module else {}
    outputs = _get_relay_input_vars(
        graph, input_shapes, prelude, default_dtype=default_dtype, is_module=is_module
    )
    param_vars, tensors, packed_param_map = convert_params(graph, params)
    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    outputs.update(param_vars)
    ret_name = _get_input_names(graph.return_node())

    # For quantized models
    if "aten::quantize_per_tensor" in op_names:
        weight_quant_params = qnn_torch.get_weight_quant_params(script_module)
        qnn_torch.add_input_quant_params_to_op_inputs(graph)
        qnn_torch.add_quant_params_to_outputs(outputs, packed_param_map, weight_quant_params)
        qnn_torch.add_quant_params(tvm_params, weight_quant_params)
        convert_map.update(qnn_torch.convert_map)

    ret = convert_operators(
        _get_operator_nodes(graph.nodes()),
        outputs,
        ret_name,
        convert_map,
        prelude,
        default_dtype=default_dtype,
    )

    mod["main"] = tvm.relay.Function(_analysis.free_vars(ret[0]), ret[0])

    return transform.RemoveUnusedFunctions()(mod), tvm_params
