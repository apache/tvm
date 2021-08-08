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
# pylint: disable=invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
"""Tensorflow2.x to relay converter ops and helper"""
import tvm
from tvm.relay.prelude import StaticTensorArrayOps, get_tensor_array_shape

from .. import op as _op
from ..ty import Any
from .common import infer_value as _infer_value
from .common import infer_type as _infer_type
from .tensorflow_ops import _get_more_static_shape_rank


def _infer_type_with_prelude(val, prelude):
    body = _infer_type(val, prelude.mod)
    return body.checked_type


def _need_prelude_for_shape_inference(op):
    return "TensorList" in op or "TensorArray" in op


def _tensorlist_reserve():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("element_dtype").name
        elem_shape = _infer_value(inputs[0], params, prelude.mod)
        elem_shape = tuple(elem_shape.numpy().astype("int32").flatten())

        if elem_shape or "shape" in attr:
            shape = attr["shape"] if "shape" in attr else elem_shape
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, shape)
            static_tensor_array_ops.register()
            tensor_array_constructor = static_tensor_array_ops.get_global_var("tensor_array")
            tensor_array = tensor_array_constructor(inputs[1])
        else:
            tensor_array_constructor = prelude.get_global_var("tensor_array", dtype_str)
            tensor_array = tensor_array_constructor(inputs[1])
        return tensor_array

    return _impl


def _tensorlist_set_item():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get("element_dtype").name
        input_ta = inputs[0]
        input_ta_shape = get_tensor_array_shape(input_ta, dtype_str, prelude)
        input_t_shape = _infer_type_with_prelude(inputs[2], prelude).shape
        input_rank = len(input_t_shape)

        if input_ta_shape is None:
            tensor_name = "tensor{}".format(input_rank)
            tensor_func = prelude.get_tensor_ctor(tensor_name, dtype_str)
            v = tensor_func(inputs[2])
            write_func = prelude.get_global_var("tensor_array_write", dtype_str)
            out = write_func(input_ta, inputs[1], v)
        else:
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
            static_tensor_array_ops.register()
            tensor_func = static_tensor_array_ops.get_ctor("tensor_constructor")
            v = tensor_func(inputs[2])
            # Write tensor with more static shape
            # convert shape with -1 to any()
            input_ta_shape_a = []
            for dim in input_ta_shape:
                if isinstance(dim, (int, tvm.tir.expr.IntImm)):
                    if dim < 0:
                        input_ta_shape_a.append(Any())
                    else:
                        input_ta_shape_a.append(dim)
                else:
                    input_ta_shape_a.append(dim)
            actual_shape = _get_more_static_shape_rank(input_t_shape, input_ta_shape_a)
            if actual_shape != input_ta_shape_a:
                new_shape = []
                num_any_dim = 0
                for dim in actual_shape:
                    if not isinstance(dim, int):
                        num_any_dim += 1
                    new_shape.append(dim if isinstance(dim, int) else -1)
                if num_any_dim <= 1:
                    v = tensor_func(_op.reshape(inputs[2], new_shape))
            write_func = prelude.get_global_var_static(
                "tensor_array_write", dtype_str, input_ta_shape_a
            )
            out = write_func(input_ta, inputs[1], v)
        return out

    return _impl


def _tensorlist_get_item():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_shape = get_tensor_array_shape(inputs[0], dtype_str, prelude)

        if input_shape is None:
            read_func = prelude.get_global_var("tensor_array_read", dtype_str)
            out = read_func(inputs[0], _op.take(inputs[1], tvm.relay.const(0)))
        else:
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_shape)
            static_tensor_array_ops.register()
            read_func = static_tensor_array_ops.get_global_var("tensor_array_read")
            out_tensor = read_func(inputs[0], _op.take(inputs[1], tvm.relay.const(0)))
            get_data_func = static_tensor_array_ops.get_global_var("tensor_get_data")
            out = get_data_func(out_tensor)
        return out

    return _impl


def _tensorlist_stack():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_ta_shape = get_tensor_array_shape(inputs[0], dtype_str, prelude)

        if input_ta_shape is None:
            stack_func = prelude.get_global_var("tensor_array_stack", dtype_str)
            out = stack_func(inputs[0])
        else:
            if "num_elements" in attr:
                num_elements = attr["num_elements"]
            static_tensor_array_ops = StaticTensorArrayOps(
                prelude, dtype_str, input_ta_shape, num_elements
            )
            static_tensor_array_ops.register()
            stack_func = prelude.get_global_var_static(
                "tensor_array_stack", dtype_str, input_ta_shape, num_elements
            )
            out_tensor = stack_func(inputs[0])
            out_shape = (
                (num_elements,) + input_ta_shape
                if num_elements and num_elements == 1
                else (Any(),) + input_ta_shape
            )
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, out_shape)
            static_tensor_array_ops.register()
            get_data_func = prelude.get_global_var_static("tensor_get_data", dtype_str, out_shape)
            out = get_data_func(out_tensor)

        return out

    return _impl


def _tensorlist_from_tensor():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr["element_dtype"].name
        input_ta_shape = _infer_type_with_prelude(inputs[0], prelude).shape

        if input_ta_shape is None:
            unstack_func = prelude.get_global_var("tensor_array_unstack", dtype_str)
            out = unstack_func(inputs[0])
        else:
            static_tensor_array_ops = StaticTensorArrayOps(prelude, dtype_str, input_ta_shape)
            static_tensor_array_ops.register()
            unstack_func = prelude.get_global_var_static(
                "tensor_array_unstack", dtype_str, input_ta_shape
            )
            out = unstack_func(inputs[0])
        return out

    return _impl


_convert_map = {
    "TensorListFromTensor": _tensorlist_from_tensor(),
    "TensorListGetItem": _tensorlist_get_item(),
    "TensorListReserve": _tensorlist_reserve(),
    "TensorListSetItem": _tensorlist_set_item(),
    "TensorListStack": _tensorlist_stack(),
}
