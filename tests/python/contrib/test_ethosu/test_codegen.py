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
import numpy as np
import tflite.Model

import tvm
import tensorflow as tf
from tvm import relay
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
            mod, input_data, output_data, accel_type, output_tolerance=1
        )

        # Assumes only two runtime.Modules are created -- i.e. single offload module
        imported_modules = compiled_models[0].executor_factory.lib.imported_modules
        assert len(imported_modules) == 2
        ethosu_module = imported_modules[0]

        # Verify generated C source
        get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
        cmms = get_cs(ethosu_module)
        cmms = bytes.fromhex(cmms)
        infra.print_payload(cmms)
        infra.verify_source(compiled_models, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape", [(1, 55, 55, 3), (1, 23, 32, 7)])
@pytest.mark.parametrize(
    "kernel_shape, activation",
    [((3, 3), "relu"), ((1, 2), None)],
)
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (2, 2)), ((3, 2), (1, 1))])
def test_tflite_depthwise_conv2d(
    accel_type,
    ifm_shape,
    kernel_shape,
    padding,
    strides,
    dilation,
    activation,
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def depthwise_conv2d(self, x):
                weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 1]
                weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                # The input strides to the TensorFlow API needs to be of shape 1x4
                tf_strides = [1, strides[0], strides[1], 1]
                op = tf.nn.depthwise_conv2d(
                    x, weight, strides=tf_strides, padding=padding, dilations=dilation
                )
                if activation:
                    op = tf.nn.relu(op)
                return op

        model = Model()
        concrete_func = model.depthwise_conv2d.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = partition_for_ethosu(relay_module, params)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

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
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


@pytest.mark.parametrize(
    "accel_type",
    ACCEL_TYPES,
)
@pytest.mark.parametrize("pooling_type", ["MAX", "AVG"])
@pytest.mark.parametrize("ifm_shape", [[1, 3, 4, 3], [1, 4, 5, 2]])
@pytest.mark.parametrize(
    "pool_shape, strides, activation_function, padding",
    [([1, 2], [1, 2], "NONE", "SAME"), ([2, 3], [2, 3], "RELU", "VALID")],
)
def test_ethosu_pooling(
    accel_type,
    ifm_shape,
    pooling_type,
    strides,
    pool_shape,
    activation_function,
    padding,
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                if pooling_type == "MAX":
                    op = tf.nn.max_pool(x, pool_shape, strides, padding)
                elif pooling_type == "AVG":
                    op = tf.nn.avg_pool(x, pool_shape, strides, padding)
                if activation_function == "RELU":
                    op = tf.nn.relu(op)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"x": ifm_shape},
        dtype_dict={"x": dtype},
    )
    mod = partition_for_ethosu(relay_module, params)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

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
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("operator_type", ["ADD", "SUB", "MUL", "MIN", "MAX"])
@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 1, 1, 1]),
        ([1, 1, 1, 1], [1, 2, 3, 4]),
        ([1, 4, 4], [4, 1]),
    ],
)
@pytest.mark.parametrize("activation_function", ["NONE", "RELU"])
def test_ethosu_binary_elementwise(
    accel_type,
    operator_type,
    ifm_shape,
    ifm2_shape,
    activation_function,
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, lhs, rhs):
                if operator_type == "ADD":
                    op = tf.math.add(lhs, rhs)
                elif operator_type == "SUB":
                    op = tf.math.subtract(lhs, rhs)
                elif operator_type == "MUL":
                    op = tf.math.multiply(lhs, rhs)
                elif operator_type == "MIN":
                    op = tf.math.minimum(lhs, rhs)
                elif operator_type == "MAX":
                    op = tf.math.maximum(lhs, rhs)
                if activation_function == "RELU":
                    op = tf.nn.relu(op)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32), tf.TensorSpec(ifm2_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                data2 = np.random.rand(*tuple(ifm2_shape)) * 2
                yield [data.astype(np.float32), data2.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"ifm": ifm_shape, "ifm2": ifm2_shape},
        dtype_dict={"ifm": dtype, "ifm2": dtype},
    )
    mod = partition_for_ethosu(mod, params)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

    compiled_models = infra.build_source(
        mod,
        input_data,
        output_data,
        accel_type,
        output_tolerance=1 if operator_type == "MAX" else 0,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    imported_modules = compiled_models[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape",
    [
        ([4], [4]),
        ([4], [1, 2, 3, 4]),
        ([1, 4, 4], [4, 1]),
    ],
)
def test_binary_add_with_non_4d_shapes(
    accel_type,
    ifm_shape,
    ifm2_shape,
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, lhs, rhs):
                return tf.math.add(lhs, rhs)

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32), tf.TensorSpec(ifm2_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                data2 = np.random.rand(*tuple(ifm2_shape)) * 2
                yield [data.astype(np.float32), data2.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"ifm": ifm_shape, "ifm2": ifm2_shape},
        dtype_dict={"ifm": dtype, "ifm2": dtype},
    )
    mod = partition_for_ethosu(mod, params)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

    compiled_models = infra.build_source(
        mod,
        input_data,
        output_data,
        accel_type,
        output_tolerance=0,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    imported_modules = compiled_models[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_binary_add_from_constant_scalar(accel_type):
    dtype = "uint8"
    ifm_shape = (1, 4, 4, 8)

    def create_relay_graph():
        inp = relay.var("input", shape=ifm_shape, dtype=dtype)
        scalar = relay.const(np.ones((1, 1, 1, 1), dtype=dtype), dtype=dtype)
        add = relay.qnn.op.add(
            inp,
            scalar,
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
        )
        func = relay.Function(relay.analysis.free_vars(add), add)
        return tvm.IRModule.from_expr(func)

    mod = create_relay_graph()
    partitioned_mod = partition_for_ethosu(mod)

    # Generate reference data
    input_data = {"input": np.random.randint(low=0, high=255, size=ifm_shape, dtype=dtype)}
    output_data = generate_ref_data(mod, input_data)

    compiled_models = infra.build_source(
        partitioned_mod,
        input_data,
        output_data,
        accel_type,
        output_tolerance=0,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    imported_modules = compiled_models[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 1, 3, 1]),
        ([1, 1, 3, 1], [1, 2, 3, 4]),
    ],
)
def test_ethosu_left_shift_binary_elemwise(
    accel_type,
    ifm_shape,
    ifm2_shape,
):
    dtype = "int32"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
        ifm2 = relay.var("ifm2", shape=ifm2_shape, dtype=dtype)
        c1 = relay.left_shift(ifm, ifm2)
        f = relay.Function([ifm, ifm2], c1)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod

    relay_mod = create_model()
    mod = partition_for_ethosu(relay_mod)

    # Generate reference data
    in_min, in_max = util.get_range_for_dtype_str(dtype)
    input_data = {
        "ifm": np.random.randint(in_min, high=in_max, size=ifm_shape, dtype=dtype),
        "ifm2": np.random.randint(0, high=32, size=ifm2_shape, dtype=dtype),
    }
    output_data = generate_ref_data(relay_mod, input_data)

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
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape, reversed_operands, ofm_dtype",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], False, "int8"),
        ([1, 2, 3, 1], [1, 1, 3, 1], False, "int32"),
        ([1, 1, 3, 1], [1, 2, 3, 1], True, "int32"),
    ],
)
def test_ethosu_right_shift_binary_elemwise(
    ifm_shape, ifm2_shape, reversed_operands, accel_type, ofm_dtype
):
    dtype = "int32"

    def create_model():
        ifm_count = int(np.prod(ifm_shape))
        ifm2_count = int(np.prod(ifm2_shape))

        # Create a "partitioned" Relay function
        ifms = relay.var("ifms", shape=[ifm_count + ifm2_count], dtype=dtype)
        split = relay.split(ifms, [ifm_count])
        ifm = relay.reshape(split[0], newshape=ifm_shape)
        ifm2 = relay.reshape(split[1], newshape=ifm2_shape)
        shr_op = infra.make_ethosu_binary_elementwise(
            ifm, ifm2, ifm_shape[3], ifm2_shape[3], "SHR", ofm_dtype, reversed_operands
        )

        glb_ethosu = relay.GlobalVar("tvmgen_default_ethos_u_main_0")
        func = (
            relay.Function([ifms], shr_op)
            .with_attr("Inline", 1)
            .with_attr("Compiler", "ethos-u")
            .with_attr("global_symbol", "tvmgen_default_ethos_u_main_0")
            .with_attr("Primitive", 1)
        )
        mod = tvm.IRModule()
        mod[glb_ethosu] = func
        mod = relay.transform.InferType()(mod)

        # Main
        ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
        ifm2 = relay.var("ifm2", shape=ifm2_shape, dtype=dtype)
        call = relay.Call(
            glb_ethosu,
            [
                relay.concatenate(
                    data=(
                        relay.reshape(ifm, newshape=ifm_count),
                        relay.reshape(ifm2, newshape=ifm2_count),
                    ),
                    axis=0,
                )
            ],
        )
        mod["main"] = relay.Function([ifm, ifm2], call)
        mod = relay.transform.InferType()(mod)
        return mod

    mod = create_model()

    # Generate reference data
    in_min, in_max = util.get_range_for_dtype_str(dtype)
    in_min, in_max = 18, 19
    lhs = np.random.randint(in_min, high=in_max, size=ifm_shape, dtype=dtype)
    rhs = np.random.randint(1, high=2, size=ifm2_shape, dtype=dtype)
    input_data = {
        "ifm": lhs,
        "ifm2": rhs,
    }

    if reversed_operands:
        lhs = np.broadcast_to(lhs, ifm2_shape)
        lhs, rhs = rhs, lhs
    else:
        rhs = np.broadcast_to(rhs, ifm_shape)

    def rounding_right_shift(lhs, rhs):
        r = 1 << (rhs - 1)
        return (lhs + r) >> rhs

    output_data = np.array(
        [rounding_right_shift(x[0], x[1]) for x in zip(lhs.flat, rhs.flat)]
    ).astype(ofm_dtype)

    compiled_model = infra.build_source(mod, input_data, [output_data], accel_type)
    imported_modules = compiled_model[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_model, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape", [(3, 2), (1, 15, 11, 7), (3, 1, 12), (400,)])
@pytest.mark.parametrize("ifm_scale, ifm_zp, ofm_scale, ofm_zp", [(1, 0, 1, 0), (0.015, 3, 0.2, 5)])
def test_ethosu_identity_codegen(ifm_shape, ifm_scale, ifm_zp, ofm_scale, ofm_zp, accel_type):
    # Create a "partitioned" Relay function
    ifm0 = relay.var("ifm0", shape=ifm_shape, dtype="int8")
    identity = infra.make_ethosu_identity(
        ifm0, ifm_scale=ifm_scale, ifm_zero_point=ifm_zp, ofm_scale=ofm_scale, ofm_zero_point=ofm_zp
    )
    mod = infra.make_partitioned_function(identity)

    in_data = np.random.randint(-120, high=120, size=ifm_shape, dtype="int8")
    requant_data = (ifm_scale * (in_data - ifm_zp)) / ofm_scale + ofm_zp
    out_data = np.round(np.clip(requant_data, -128, 127)).astype("int8")

    compiled_model = infra.build_source(
        mod, {"ifm": in_data}, [out_data], accel_type, output_tolerance=1
    )

    imported_modules = compiled_model[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_model, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, new_shape",
    [
        ((1, 4, 1, 2), (1, 1, 1, 8)),
        ((12, 20), (1, 6, 4, 10)),
        ((12, 20), (6, 4, 10)),
        ((20,), (4, 5)),
        ((12, 2, 10), (0, -3)),
        ((11, 3, 25), (-1,)),
        ((8, 7, 3), (-4, 1, 8, -2)),
    ],
)
def test_relay_reshape_codegen(ifm_shape, new_shape, accel_type):
    # Create a "partitioned" Relay graph
    ifm0 = relay.var("ifm0", shape=ifm_shape, dtype="int8")
    reshape = relay.op.reshape(ifm0, newshape=new_shape)
    mod = infra.make_partitioned_function(reshape)

    data = np.random.randint(-128, high=127, size=ifm_shape, dtype="int8")

    # Generate a reference output using Relay reshape that doesn't get offloaded
    ref_mod = tvm.IRModule()
    ref_mod["main"] = relay.Function([ifm0], reshape)
    ref_mod = relay.transform.InferType()(ref_mod)

    out_data = generate_ref_data(ref_mod, {"ifm0": data})

    compiled_model = infra.build_source(
        mod,
        {"ifm": data},
        out_data,
        accel_type,
    )

    imported_modules = compiled_model[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_model, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, begin, end",
    [
        ([1, 10, 50, 4], [0, 5, 11, 2], [1, 10, 22, 3]),
        ([15, 17, 3], [3, 0, 1], [11, 17, 3]),
        ([7, 6043], [0, 704], [1, 3564]),
        ([5000], [123], [2274]),
    ],
)
def test_relay_strided_slice_codegen(ifm_shape, begin, end, accel_type):
    # Create a "partitioned" Relay graph
    ifm0 = relay.var("ifm0", shape=ifm_shape, dtype="int8")
    strided_slice = relay.op.strided_slice(ifm0, begin, end)
    mod = infra.make_partitioned_function(strided_slice)

    input_data = np.random.randint(-128, high=127, size=ifm_shape, dtype="int8")

    # Generate a reference output using Relay strided slice that doesn't get offloaded
    ref_mod = tvm.IRModule()
    ref_mod["main"] = relay.Function([ifm0], strided_slice)
    ref_mod = relay.transform.InferType()(ref_mod)

    out_data = generate_ref_data(ref_mod, {"ifm0": input_data})

    compiled_model = infra.build_source(
        mod,
        {"ifm": input_data},
        out_data,
        accel_type,
    )

    imported_modules = compiled_model[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_model, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("operator_type", ["ABS"])
@pytest.mark.parametrize(
    "ifm_shape",
    [[1, 5, 12, 4], [1, 1, 2], [4, 3, 2], [10, 20], [345]],
)
def test_ethosu_unary_elementwise(
    accel_type,
    operator_type,
    ifm_shape,
):
    dtype = "int8"

    def get_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def abs_func(self, x):
                if operator_type == "ABS":
                    op = tf.math.abs(x)
                return op

        model = Model()

        concrete_func = model.abs_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32) * 2 - 1]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = get_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = partition_for_ethosu(relay_module, params)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

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
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


def test_ethosu_section_name():
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

    accel_type = "ethos-u55-256"
    relay_module, _ = create_graph_single("input", (1, 300, 300, 3), "int8")
    input_dtype = "int8"
    mod = partition_for_ethosu(relay_module)

    # Generate reference data
    in_min, in_max = util.get_range_for_dtype_str(input_dtype)
    input_data = {
        "input": np.random.randint(in_min, high=in_max, size=(1, 300, 300, 3), dtype=input_dtype)
    }
    output_data = generate_ref_data(relay_module, input_data)

    compiled_models = infra.build_source(
        mod, input_data, output_data, accel_type, output_tolerance=1
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    imported_modules = compiled_models[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    source = ethosu_module.get_source()
    assert (
        '__attribute__((section(".rodata.tvm"), aligned(16))) static int8_t cms_data_data' in source
    )
    assert '__attribute__((section(".rodata.tvm"), aligned(16))) static int8_t weights' in source


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_ethosu_clz(accel_type):
    ifm_shape = (1, 42, 5, 4)
    # Create a "partitioned" Relay function
    ifm0 = relay.var("ifm0", shape=ifm_shape, dtype="int32")
    clz = infra.make_ethosu_unary_elementwise(ifm0, 4, "CLZ")
    mod = infra.make_partitioned_function(clz)

    in_data = np.random.randint(-500000, high=500000, size=ifm_shape, dtype="int32")

    def clz_comp(n):
        n_bin = np.binary_repr(n)
        if n_bin[0] == "-":
            return 0
        else:
            return 32 - len(n_bin)

    out_data = np.array([clz_comp(i) for i in in_data.ravel()]).reshape(ifm_shape).astype("int32")

    compiled_model = infra.build_source(mod, {"ifm": in_data}, [out_data], accel_type)

    imported_modules = compiled_model[0].executor_factory.lib.imported_modules
    assert len(imported_modules) == 2
    ethosu_module = imported_modules[0]

    # Verify generated C source
    get_cs = tvm._ffi.get_global_func("runtime.module.ethos-u.getcs")
    cmms = get_cs(ethosu_module)
    cmms = bytes.fromhex(cmms)

    infra.print_payload(cmms)
    infra.verify_source(compiled_model, accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
