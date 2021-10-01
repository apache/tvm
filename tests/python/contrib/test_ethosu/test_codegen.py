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
# pylint: disable=invalid-name, unused-argument
import pytest

pytest.importorskip("ethosu.vela")
import os
import numpy as np
import pathlib

import tvm
import tvm.micro as micro
from tvm import relay
from tvm.relay.backend.contrib import ethosu
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tests.python.relay.aot.aot_test_utils import generate_ref_data

from . import relay_ir_builder
from . import infra

ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32"]


def infer_type_function_pass(func):
    mod = tvm.IRModule()
    mod["test"] = func
    mod = relay.transform.InferType()(mod)
    return mod["test"]


def get_shape_expr(in_expr, out_expr):
    main_f = relay.Function([in_expr], out_expr)
    main_f = infer_type_function_pass(main_f)
    shape = [int(i) for i in main_f.body.checked_type.shape]
    return shape


@pytest.mark.parametrize(
    "accel_type",
    ACCEL_TYPES,
)
def test_ethosu_conv2d(accel_type):
    def create_graph_single(input_tensor_name, input_tensor_shape, input_tensor_dtype):
        c1_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c1_params.ifm.shape = input_tensor_shape
        c1_params.kernel.shape = (3, 3, c1_params.ifm.shape[3], 32)
        c1_params.kernel.sc = relay.const(np.random.rand(32) * 2, "float32")
        c1_params.strides = (1, 1)
        c1_params.pad = "VALID"
        c1_params.update_output_qnn_params(
            input_tensor_dtype, input_tensor_dtype, input_tensor_dtype
        )
        input0 = relay.var(input_tensor_name, shape=c1_params.ifm.shape, dtype=c1_params.ifm.dtype)
        c1, new_params = relay_ir_builder.create_qnn_conv2d(c1_params, input0)
        c1_params.ofm.shape = get_shape_expr(input0, c1)

        f = relay.Function([input0], c1)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod, [c1_params]

    def create_graph_double(input_tensor_name, input_tensor_shape, input_tensor_dtype):
        c1_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c1_params.ifm.shape = input_tensor_shape
        c1_params.kernel.shape = (7, 7, c1_params.ifm.shape[3], 8)
        c1_params.strides = (2, 2)
        c1_params.pad = "VALID"
        c1_params.update_output_qnn_params(
            input_tensor_dtype, input_tensor_dtype, input_tensor_dtype
        )
        input0 = relay.var(input_tensor_name, shape=c1_params.ifm.shape, dtype=c1_params.ifm.dtype)
        c1, new_params = relay_ir_builder.create_qnn_conv2d(c1_params, input0)
        c1_params.ofm.shape = get_shape_expr(input0, c1)

        c2_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c2_params.ifm.shape = c1_params.ofm.shape
        c2_params.kernel.shape = (5, 5, c2_params.ifm.shape[3], 16)
        c2_params.strides = (1, 1)
        c2_params.pad = "SAME"
        c2_params.update_output_qnn_params()
        c2, new_params = relay_ir_builder.create_qnn_conv2d(c2_params, c1)
        c2_params.ofm.shape = get_shape_expr(input0, c2)

        f = relay.Function([input0], c2)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod, [c2_params, c1_params]

    def create_graph_activation(input_tensor_name, input_tensor_shape, input_tensor_dtype):
        c1_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c1_params.ifm.shape = input_tensor_shape
        c1_params.kernel.shape = (7, 7, c1_params.ifm.shape[3], 8)
        c1_params.strides = (2, 2)
        c1_params.pad = "VALID"
        c1_params.activation = "CLIP"
        c1_params.clip_min = 90
        c1_params.clip_max = 110
        c1_params.update_output_qnn_params(
            input_tensor_dtype, input_tensor_dtype, input_tensor_dtype
        )
        input0 = relay.var(input_tensor_name, shape=c1_params.ifm.shape, dtype=c1_params.ifm.dtype)
        c1, new_params = relay_ir_builder.create_qnn_conv2d(c1_params, input0)
        c1_params.ofm.shape = get_shape_expr(input0, c1)

        c2_params = relay_ir_builder.QnnConv2DParams(input_tensor_dtype)
        c2_params.ifm.shape = c1_params.ofm.shape
        c2_params.kernel.shape = (5, 5, c2_params.ifm.shape[3], 16)
        c2_params.strides = (1, 1)
        c2_params.pad = "SAME"
        c2_params.update_output_qnn_params()
        c2, new_params = relay_ir_builder.create_qnn_conv2d(c2_params, c1)
        c2_params.ofm.shape = get_shape_expr(input0, c2)

        f = relay.Function([input0], c2)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod, [c2_params, c1_params]

    test_cases = [
        (create_graph_single, ["input", (1, 300, 300, 3), "int8"]),
        (create_graph_double, ["input", (1, 128, 256, 4), "int8"]),
        (create_graph_activation, ["input", (1, 64, 100, 4), "int8"]),
    ]
    np.random.seed(42)
    for test_case in test_cases:
        relay_module, conv_params = test_case[0](*test_case[1])
        input_tensor, input_shape, input_dtype = test_case[1]
        mod = partition_for_ethosu(relay_module)

        # Generate reference data
        in_min, in_max = util.get_range_for_dtype_str(input_dtype)
        input_data = {
            input_tensor: np.random.randint(
                in_min, high=in_max, size=input_shape, dtype=input_dtype
            )
        }
        output_data = generate_ref_data(relay_module, input_data)

        compiled_models = infra.build_source(
            mod,
            input_data,
            output_data,
            accel_type,
        )

        # Assumes only two runtime.Modules are created -- i.e. single offload module
        imported_modules = compiled_models[0].executor_factory.lib.imported_modules
        assert len(imported_modules) == 2
        ethosu_module = imported_modules[0]

        # Verify generated C source
        get_cs = tvm._ffi.get_global_func("runtime.module.ethosu.getcs")
        cmms = get_cs(ethosu_module)
        cmms = bytes.fromhex(cmms)
        infra.print_payload(cmms)
        infra.verify_source(compiled_models, accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
