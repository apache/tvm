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
from tvm.testing.aot import generate_ref_data

from . import infra


ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


def is_u55_accel_type(accel_type):
    return "u55" in accel_type


@pytest.mark.parametrize("accel_type", ACCEL_TYPES + ["ethos-u65-512"])
@pytest.mark.parametrize("ifm_shape", [(1, 299, 299, 2), (1, 55, 55, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 2), (1, 3)])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (2, 1)), ((3, 2), (1, 1))])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("activation", ["NONE", "RELU"])
def test_ethosu_conv2d_single(
    ifm_shape,
    kernel_shape,
    strides,
    dilation,
    padding,
    accel_type,
    activation,
):
    np.random.seed(0)

    @tf.function
    def conv2d(x):
        # Use tf.nn API to create the model
        tf_strides = [1, strides[0], strides[1], 1]
        op = tf.nn.conv2d(
            x,
            filters=tf.constant(
                np.random.uniform(size=[kernel_shape[0], kernel_shape[1], ifm_shape[3], 3]),
                dtype=tf.float32,
            ),
            strides=tf_strides,
            padding=padding,
            dilations=dilation,
        )
        if activation == "RELU":
            op = tf.nn.relu(op)
        return op

    infra.compare_tvm_with_tflite(conv2d, [ifm_shape], accel_type)


def test_tflite_conv2d_with_separate_pad():
    np.random.seed(0)

    ifm_shape = (1, 55, 34, 3)
    kernel_shape = (3, 2)
    strides = (1, 1)
    dilation = (2, 1)
    padding = (0, 0, 1, 1)

    @tf.function
    def conv2d(x):
        tf_strides = [1, strides[0], strides[1], 1]
        op = tf.pad(
            x,
            [[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]],
            "CONSTANT",
        )
        weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 3]
        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        return tf.nn.conv2d(
            op,
            weight,
            strides=tf_strides,
            padding="VALID",
            dilations=dilation,
        )

    infra.compare_tvm_with_tflite(conv2d, [ifm_shape], "ethos-u55-256")


@pytest.mark.parametrize("ifm_shape", [(1, 214, 227, 2), (1, 27, 42, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 2), (1, 3)])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (2, 1)), ((3, 2), (1, 1))])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("accel_type", ACCEL_TYPES + ["ethos-u65-512"])
@pytest.mark.parametrize("activation", ["NONE", "RELU"])
def test_ethosu_conv2d_double(
    ifm_shape,
    kernel_shape,
    strides,
    dilation,
    padding,
    accel_type,
    activation,
):
    np.random.seed(0)

    @tf.function
    def conv2d_double(x):
        # Use tf.nn API to create the model with two convolutions
        op = tf.nn.conv2d(
            x,
            filters=tf.constant(
                np.random.uniform(size=[kernel_shape[0], kernel_shape[1], ifm_shape[3], 5]),
                dtype=tf.float32,
            ),
            strides=strides,
            padding=padding,
            dilations=dilation,
        )
        # Second convolution
        op2 = tf.nn.conv2d(
            op,
            filters=tf.constant(
                np.random.uniform(size=(kernel_shape[0], kernel_shape[1], 5, 3)),
                dtype=tf.float32,
            ),
            strides=strides,
            padding=padding,
            dilations=dilation,
        )
        if activation == "RELU":
            op2 = tf.nn.relu(op2)
        return op2

    infra.compare_tvm_with_tflite(conv2d_double, [ifm_shape], accel_type)


@pytest.mark.parametrize("weight_min, weight_max", [(0.0, 1e-11), (-1e10, 1e10)])
def test_out_of_range_scaling(weight_min, weight_max):
    np.random.seed(0)
    ifm_shape = (1, 6, 6, 2)
    strides = (1, 1)
    kernel_shape = (1, 1)
    dilation = (1, 1)
    padding = "SAME"
    activation = "RELU"
    accel_type = "ethos-u55-128"

    @tf.function
    def conv_invalid_scale(x):
        # Use tf.nn API to create the model
        tf_strides = [1, strides[0], strides[1], 1]
        weights = np.random.uniform(size=[kernel_shape[0], kernel_shape[1], 2, 2])
        # Overwrite to force quantization that produces out of range shift values
        weights[0][0][0][0] = weight_min
        weights[0][0][1][0] = weight_max
        op = tf.nn.conv2d(
            x,
            filters=tf.constant(
                weights,
                dtype=tf.float32,
            ),
            strides=tf_strides,
            padding=padding,
            dilations=dilation,
        )
        if activation == "RELU":
            op = tf.nn.relu(op)
        return op

    infra.compare_tvm_with_tflite(conv_invalid_scale, [ifm_shape], accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape", [(1, 55, 55, 3), (1, 23, 32, 7)])
@pytest.mark.parametrize(
    "kernel_shape, activation_function",
    [((3, 3), "RELU"), ((1, 2), "NONE")],
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
    activation_function,
):
    np.random.seed(0)

    @tf.function
    def depthwise_conv2d(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 1]
        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        # The input strides to the TensorFlow API needs to be of shape 1x4
        tf_strides = [1, strides[0], strides[1], 1]
        op = tf.nn.depthwise_conv2d(
            x, weight, strides=tf_strides, padding=padding, dilations=dilation
        )
        if activation_function == "RELU":
            op = tf.nn.relu(op)
        return op

    infra.compare_tvm_with_tflite(depthwise_conv2d, [ifm_shape], accel_type)


def test_tflite_depthwise_conv2d_with_separate_pad():
    np.random.seed(0)

    ifm_shape = (1, 23, 32, 7)
    kernel_shape = (1, 2)
    strides = (3, 2)
    dilation = (1, 1)
    padding = (0, 0, 1, 1)

    @tf.function
    def depthwise_conv2d(x):
        tf_strides = [1, strides[0], strides[1], 1]
        op = tf.pad(
            x,
            [[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]],
            "CONSTANT",
        )
        weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 1]
        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        return tf.nn.depthwise_conv2d(
            op,
            weight,
            strides=tf_strides,
            padding="VALID",
            dilations=dilation,
        )

    infra.compare_tvm_with_tflite(depthwise_conv2d, [ifm_shape], "ethos-u55-256")


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
    np.random.seed(0)

    @tf.function
    def pooling(x):
        if pooling_type == "MAX":
            op = tf.nn.max_pool(x, pool_shape, strides, padding)
        elif pooling_type == "AVG":
            op = tf.nn.avg_pool(x, pool_shape, strides, padding)
        if activation_function == "RELU":
            op = tf.nn.relu(op)
        return op

    infra.compare_tvm_with_tflite(pooling, [ifm_shape], accel_type)


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
    np.random.seed(0)

    @tf.function
    def binary_elementwise(lhs, rhs):
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

    infra.compare_tvm_with_tflite(
        binary_elementwise,
        shapes=[ifm_shape, ifm2_shape],
        ranges=[(0, 1), (0, 2)],
        accel_type=accel_type,
        enable_cascader=is_u55_accel_type(accel_type),
    )


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
    request,
    accel_type,
    ifm_shape,
    ifm2_shape,
):
    np.random.seed(0)

    @tf.function
    def binary_elementwise(lhs, rhs):
        return tf.math.add(lhs, rhs)

    infra.compare_tvm_with_tflite(
        binary_elementwise,
        shapes=[ifm_shape, ifm2_shape],
        ranges=[(0, 1), (0, 2)],
        accel_type=accel_type,
        enable_cascader=is_u55_accel_type(accel_type),
    )


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/12634")
@pytest.mark.parametrize(
    "accel_type",
    ACCEL_TYPES,
)
@pytest.mark.parametrize(
    "ifm_shape, axis, keep_dims, use_same_quantization",
    [
        # mean to depthwise + multiply
        [(1, 8, 16, 16), (1, 2), True, False],
        [(1, 3, 4), (0, 1), True, False],
        [(1, 65, 2, 1), (1, 2), True, False],  # special case when h > 64
        # mean to average pool
        [(1, 8, 16, 16), (2,), False, True],
        [(3, 3, 4), (0,), True, True],
        [(8, 5), (0,), False, True],
        # mean to depthwise
        [(1, 8, 16, 16), (2,), True, False],
        [(1, 8, 16, 16), (2, 1), False, False],
        [(8, 4), (0,), False, False],
    ],
)
def test_mean(accel_type, ifm_shape, axis, keep_dims, use_same_quantization):
    np.random.seed(0)
    dtype = "int8"

    def create_mod_from_tflite():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                op = tf.math.reduce_mean(x, axis=axis, keepdims=keep_dims)
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
        tflite_graph = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"ifm": ifm_shape},
            dtype_dict={"ifm": dtype},
        )
        input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)
        return mod, input_data, output_data

    def create_mod_from_relay():
        ifm = relay.var("input", shape=ifm_shape, dtype=dtype)
        cast = relay.cast(ifm, dtype="int32")
        mean = relay.mean(cast, axis=axis, keepdims=keep_dims)
        requantize = relay.qnn.op.requantize(
            mean,
            input_scale=relay.const(1.0, dtype="float32"),
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(0, dtype="int32"),
        )

        func = relay.Function(relay.analysis.free_vars(requantize), requantize)
        mod = tvm.IRModule.from_expr(func)

        input_data = {"input": np.random.randint(low=-127, high=128, size=ifm_shape, dtype=dtype)}
        output_data = generate_ref_data(mod, input_data)
        return mod, input_data, output_data

    mod, input_data, output_data = (
        create_mod_from_relay() if use_same_quantization else create_mod_from_tflite()
    )
    mod = partition_for_ethosu(mod)

    test_runner = infra.create_test_runner(accel_type)
    compiled_models = infra.build_source(mod, input_data, output_data, test_runner)

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    get_artifacts = tvm._ffi.get_global_func("runtime.module.ethos-u.get_artifacts")
    compilation_artifacts = get_artifacts(ethosu_module)
    cmms = bytes.fromhex(compilation_artifacts[0].command_stream)
    infra.print_payload(cmms)
    infra.verify_source(compiled_models, test_runner)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("constant", [np.ones((1, 1, 1, 1)), np.array(1)])
def test_elementwise_add_from_constant_scalar(accel_type, dtype, constant):
    np.random.seed(0)
    ifm_shape = (1, 4, 4, 8)

    def create_relay_graph():
        inp = relay.var("input", shape=ifm_shape, dtype=dtype)
        scalar = relay.const(constant, dtype=dtype)
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
        return tvm.IRModule.from_expr(relay.Function(relay.analysis.free_vars(add), add))

    cpu_mod = create_relay_graph()
    ethosu_mod = partition_for_ethosu(cpu_mod)

    # Generate reference data
    input_data = {
        "input": np.random.randint(
            low=np.iinfo(dtype).min, high=np.iinfo(dtype).max, size=ifm_shape, dtype=dtype
        ),
    }
    output_data = generate_ref_data(cpu_mod, input_data)

    # Scalar constants are not supported by the cascader
    infra.compare_ethosu_with_reference(
        ethosu_mod, input_data, output_data, accel_type, enable_cascader=False
    )


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
    np.random.seed(0)
    dtype = "int32"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
        ifm2 = relay.var("ifm2", shape=ifm2_shape, dtype=dtype)
        c1 = relay.left_shift(ifm, ifm2)
        return tvm.IRModule.from_expr(relay.Function([ifm, ifm2], c1))

    cpu_mod = create_model()

    # Generate reference data
    in_min, in_max = util.get_range_for_dtype_str(dtype)
    input_data = {
        "ifm": np.random.randint(in_min, high=in_max, size=ifm_shape, dtype=dtype),
        "ifm2": np.random.randint(0, high=32, size=ifm2_shape, dtype=dtype),
    }
    output_data = generate_ref_data(cpu_mod, input_data)
    ethosu_mod = partition_for_ethosu(cpu_mod)

    infra.compare_ethosu_with_reference(ethosu_mod, input_data, output_data, accel_type)


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
    np.random.seed(0)
    dtype = "int32"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
        ifm2 = relay.var("ifm2", shape=ifm2_shape, dtype=dtype)
        shr_op = infra.make_ethosu_binary_elementwise(
            ifm, ifm2, ifm_shape[3], ifm2_shape[3], "SHR", ofm_dtype, reversed_operands
        )
        return tvm.IRModule.from_expr(relay.Function([ifm, ifm2], shr_op))

    def generate_output_data(input_data):
        lhs = input_data["ifm"]
        rhs = input_data["ifm2"]
        if reversed_operands:
            lhs = np.broadcast_to(lhs, ifm2_shape)
            lhs, rhs = rhs, lhs
        else:
            rhs = np.broadcast_to(rhs, ifm_shape)

        def rounding_right_shift(lhs, rhs):
            r = 1 << (rhs - 1)
            return (lhs + r) >> rhs

        return [
            np.array([rounding_right_shift(x[0], x[1]) for x in zip(lhs.flat, rhs.flat)]).astype(
                ofm_dtype
            )
        ]

    cpu_mod = create_model()

    # Generate reference data
    in_min, in_max = util.get_range_for_dtype_str(dtype)
    in_min, in_max = 18, 19
    lhs = np.random.randint(in_min, high=in_max, size=ifm_shape, dtype=dtype)
    rhs = np.random.randint(1, high=2, size=ifm2_shape, dtype=dtype)
    input_data = {
        "ifm": lhs,
        "ifm2": rhs,
    }
    output_data = {"output": generate_output_data(input_data)[0]}
    ethosu_mod = infra.create_ethosu_partition(cpu_mod)

    infra.compare_ethosu_with_reference(ethosu_mod, input_data, output_data, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape", [(3, 2), (1, 15, 11, 7), (3, 1, 12), (400,)])
@pytest.mark.parametrize("ifm_scale, ifm_zp, ofm_scale, ofm_zp", [(1, 0, 1, 0), (0.015, 3, 0.2, 5)])
def test_ethosu_identity_codegen(
    request, ifm_shape, ifm_scale, ifm_zp, ofm_scale, ofm_zp, accel_type
):
    np.random.seed(0)

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        identity = infra.make_ethosu_identity(
            ifm,
            ifm_scale=ifm_scale,
            ifm_zero_point=ifm_zp,
            ofm_scale=ofm_scale,
            ofm_zero_point=ofm_zp,
        )
        return tvm.IRModule.from_expr(relay.Function([ifm], identity))

    def generate_output_data(input_data):
        requant_data = (ifm_scale * (input_data["ifm"] - ifm_zp)) / ofm_scale + ofm_zp
        return [np.round(np.clip(requant_data, -128, 127)).astype("int8")]

    cpu_mod = create_model()
    input_data = {"ifm": np.random.randint(-120, high=120, size=ifm_shape, dtype="int8")}
    output_data = {"output": generate_output_data(input_data)[0]}
    ethosu_mod = infra.create_ethosu_partition(cpu_mod)

    infra.compare_ethosu_with_reference(
        ethosu_mod,
        input_data,
        output_data,
        accel_type,
        output_tolerance=1,
        enable_cascader=is_u55_accel_type(accel_type),
    )


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
    np.random.seed(0)

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        reshape = relay.op.reshape(ifm, newshape=new_shape)
        return tvm.IRModule.from_expr(relay.Function([ifm], reshape))

    cpu_mod = create_model()
    input_data = {"ifm": np.random.randint(-128, high=127, size=ifm_shape, dtype="int8")}
    output_data = generate_ref_data(cpu_mod, input_data)
    ethosu_mod = infra.create_ethosu_partition(cpu_mod)

    infra.compare_ethosu_with_reference(
        ethosu_mod,
        input_data,
        output_data,
        accel_type,
        enable_cascader=is_u55_accel_type(accel_type),
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, begin, size",
    [
        ([1, 10, 50, 4], [0, 5, 11, 2], [1, 5, 11, 1]),
        ([15, 17, 3], [3, 0, 1], [8, 17, 2]),
        ([7, 6043], [0, 704], [1, 2860]),
        ([5000], [123], [2151]),
    ],
)
def test_tflite_slice(request, accel_type, ifm_shape, begin, size):
    np.random.seed(0)

    @tf.function
    def slice_func(x):
        return tf.slice(x, begin, size)

    infra.compare_tvm_with_tflite(
        slice_func, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, begin, end",
    [([1, 1, 5, 8], [0, 0, 0, 0], [1, 1, 2, 3]), ([1, 3, 3], [0, 1, 2], [1, 2, 3])],
)
def test_tflite_strided_slice(accel_type, ifm_shape, begin, end):
    np.random.seed(0)

    @tf.function
    def strided_slice_func(x):
        return tf.strided_slice(x, begin, end)

    infra.compare_tvm_with_tflite(
        strided_slice_func, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("operator_type", ["ABS"])
@pytest.mark.parametrize(
    "ifm_shape",
    [[1, 5, 12, 4], [1, 1, 2], [4, 3, 2], [10, 20], [345]],
)
def test_ethosu_unary_elementwise(
    request,
    accel_type,
    operator_type,
    ifm_shape,
):
    np.random.seed(0)

    @tf.function
    def abs_func(x):
        if operator_type == "ABS":
            op = tf.math.abs(x)
        return op

    infra.compare_tvm_with_tflite(
        abs_func,
        [ifm_shape],
        accel_type,
        enable_cascader=is_u55_accel_type(accel_type),
    )


def test_ethosu_section_name():
    np.random.seed(0)

    @tf.function
    def depthwise_conv2d(x):
        weight_shape = [3, 3, 3, 1]
        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        tf_strides = [1, 1, 1, 1]
        op = tf.nn.depthwise_conv2d(x, weight, strides=tf_strides, padding="SAME", dilations=(2, 2))
        return op

    mod, tflite_graph = infra.get_tflite_graph(depthwise_conv2d, [(1, 55, 55, 3)])

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

    test_runner = infra.create_test_runner()
    compiled_models = infra.build_source(mod, input_data, output_data, test_runner)

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    source = ethosu_module.get_source()
    assert (
        '__attribute__((section(".rodata.tvm"), aligned(16))) static int8_t tvmgen_default_ethos_u_main_0_cms_data_data'
        in source
    )
    assert (
        '__attribute__((section(".rodata.tvm"), aligned(16))) static int8_t tvmgen_default_ethos_u_main_0_weights'
        in source
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_ethosu_clz(accel_type):
    np.random.seed(0)
    ifm_shape = (1, 42, 5, 4)

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int32")
        clz = infra.make_ethosu_unary_elementwise(ifm, 4, "CLZ")
        return tvm.IRModule.from_expr(relay.Function([ifm], clz))

    def generate_output_data(input_data):
        def clz_comp(n):
            n_bin = np.binary_repr(n)
            if n_bin[0] == "-":
                return 0
            else:
                return 32 - len(n_bin)

        return [
            np.array([clz_comp(i) for i in input_data["ifm"].ravel()])
            .reshape(ifm_shape)
            .astype("int32")
        ]

    cpu_mod = create_model()
    input_data = {"ifm": np.random.randint(-500000, high=500000, size=ifm_shape, dtype="int32")}
    output_data = {"output": generate_output_data(input_data)[0]}
    ethosu_mod = infra.create_ethosu_partition(cpu_mod)

    infra.compare_ethosu_with_reference(ethosu_mod, input_data, output_data, accel_type)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_tflite_tanh(accel_type):
    np.random.seed(0)
    ifm_shape = [1, 115, 32, 7]

    @tf.function
    def tanh_func(x):
        op = tf.nn.tanh(x)
        return op

    infra.compare_tvm_with_tflite(
        tanh_func, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape", [(1, 5, 5, 3), (1, 12, 9, 1)])
def test_tflite_hard_swish(accel_type, ifm_shape):
    np.random.seed(0)

    @tf.function
    def hard_swish_func(x):
        op = tf.keras.layers.Lambda(
            lambda x: x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
        )(x)
        return op

    infra.compare_tvm_with_tflite(hard_swish_func, [ifm_shape], accel_type, ranges=[(-1, 1)])


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "shapes, axis",
    [
        ([(2, 3), (4, 3)], 0),
        ([(3, 2, 1), (3, 1, 1)], 1),
        ([(10,), (13,), (14,)], 0),
        ([(1, 5, 2, 1), (1, 5, 7, 1), (1, 5, 3, 1)], 2),
    ],
)
def test_tflite_concat(shapes, axis, accel_type):
    np.random.seed(0)

    @tf.function
    def concat_func(*inputs):
        op = tf.concat(list(inputs), axis)
        return op

    infra.compare_tvm_with_tflite(concat_func, shapes, accel_type, enable_cascader=False)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_tflite_sigmoid(accel_type):
    np.random.seed(0)
    ifm_shape = [1, 135, 41, 6]

    @tf.function
    def sigmoid_function(x):
        op = tf.nn.sigmoid(x)
        return op

    infra.compare_tvm_with_tflite(
        sigmoid_function, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


# This codegen test checks both, split and split_v
@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, num_or_size_splits, axis",
    [
        ((1, 4, 6, 8), (1, 3, 4), 3),
        ((4, 6, 8), 2, 0),
        ((50,), 25, 0),
        ((5, 11), 1, 1),
        ((13,), (13,), 0),
        ((22, 7), (4, -1), 1),
    ],
)
def test_tflite_split(accel_type, ifm_shape, num_or_size_splits, axis):
    np.random.seed(0)

    @tf.function
    def split_func(x):
        op = tf.split(x, num_or_size_splits, axis=axis)
        return op

    infra.compare_tvm_with_tflite(split_func, [ifm_shape], accel_type, enable_cascader=False)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,ifm_scale,ifm_zp,ofm_scale,ofm_zp",
    [
        [(1, 8, 8, 3), 1.0, 0, 1.0, 0],
        [(1, 20, 30, 3), 1.345, 34, 0.32, -23],
    ],
)
def test_ethosu_requantize(accel_type, ifm_shape, ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    np.random.seed(0)
    dtype = "int8"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        requantize = relay.qnn.op.requantize(
            ifm,
            relay.const(ifm_scale, dtype="float32"),
            relay.const(ifm_zp, dtype="int32"),
            relay.const(ofm_scale, dtype="float32"),
            relay.const(ofm_zp, dtype="int32"),
        )
        return tvm.IRModule.from_expr(relay.Function([ifm], requantize))

    cpu_mod = create_model()
    input_data = {"ifm": np.random.randint(-128, high=127, size=ifm_shape, dtype=dtype)}
    output_data = generate_ref_data(cpu_mod, input_data)
    ethosu_mod = partition_for_ethosu(cpu_mod)

    infra.compare_ethosu_with_reference(
        ethosu_mod,
        input_data,
        output_data,
        accel_type,
        enable_cascader=is_u55_accel_type(accel_type),
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape,axis", [((2,), 0), ((1, 3, 3), 2)])
def test_tflite_expand_dims(accel_type, ifm_shape, axis):
    np.random.seed(0)

    @tf.function
    def expand_dims_func(x):
        return tf.expand_dims(x, axis=axis)

    infra.compare_tvm_with_tflite(
        expand_dims_func, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,axis", [((1, 1, 2, 1), 0), ((1, 3, 3, 1), 3), ((1, 1, 2, 1), None)]
)
def test_tflite_squeeze(accel_type, ifm_shape, axis):
    np.random.seed(0)

    @tf.function
    def squeeze_func(x):
        return tf.squeeze(x, axis=axis)

    infra.compare_tvm_with_tflite(
        squeeze_func, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,size",
    [[(1, 2, 2, 1), (4, 4)], [(1, 4, 7, 3), (8, 14)], [(1, 3, 5, 3), (3, 5)]],
)
def test_tflite_resize2d_nearest_neighbor(accel_type, ifm_shape, size):
    np.random.seed(0)
    align_corners = False

    @tf.function
    def resize_model(x):
        return tf.compat.v1.image.resize_nearest_neighbor(
            x, size, align_corners=align_corners, half_pixel_centers=False
        )

    infra.compare_tvm_with_tflite(
        resize_model, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,size,align_corners",
    [
        [(1, 2, 2, 1), (4, 4), False],
        [(1, 4, 7, 3), (8, 14), False],
        [(1, 2, 2, 1), (3, 3), True],
        [(1, 4, 7, 3), (7, 13), True],
        [(1, 3, 5, 3), (3, 5), False],
    ],
)
def test_tflite_resize2d_bilinear(accel_type, ifm_shape, size, align_corners):
    np.random.seed(0)

    @tf.function
    def resize_model(x):
        return tf.compat.v1.image.resize_bilinear(
            x, size, align_corners=align_corners, half_pixel_centers=False
        )

    infra.compare_tvm_with_tflite(
        resize_model, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,ofm_shape,kernel_shape,padding",
    [
        [(1, 2, 2, 1), (1, 4, 4, 1), (3, 3), "SAME"],
        [(1, 2, 2, 1), (1, 9, 9, 1), (7, 7), "VALID"],
        [(1, 2, 4, 3), (1, 4, 8, 3), (5, 3), "SAME"],
        [(1, 10, 5, 3), (1, 21, 13, 3), (3, 5), "VALID"],
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_tflite_transpose_convolution(
    accel_type, ifm_shape, ofm_shape, kernel_shape, padding, has_bias
):
    np.random.seed(0)
    dilations = (1, 1)
    strides = (2, 2)

    @tf.function
    def conv2d_transpose(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], ofm_shape[3]]
        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        bias_shape = ofm_shape[3]
        bias = tf.constant(np.random.uniform(size=bias_shape), dtype=tf.float32)
        tf_strides = [1, strides[0], strides[1], 1]
        op = tf.nn.conv2d_transpose(
            x,
            weight,
            output_shape=ofm_shape,
            strides=tf_strides,
            padding=padding,
            dilations=dilations,
        )
        if has_bias:
            op = tf.nn.bias_add(op, bias)
        return op

    infra.compare_tvm_with_tflite(
        conv2d_transpose,
        [ifm_shape],
        accel_type=accel_type,
        enable_cascader=is_u55_accel_type(accel_type),
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shapes,axis",
    [
        ([(1, 2, 2), (1, 2, 2), (1, 2, 2)], 2),
        ([(5, 4), (5, 4)], 1),
        ([(1,), (1,)], 0),
        ([(3, 1), (3, 1), (3, 1), (3, 1)], 0),
    ],
)
def test_tflite_pack(accel_type, ifm_shapes, axis):
    np.random.seed(0)

    @tf.function
    def pack_func(*inputs):
        return tf.stack(inputs, axis=axis)

    infra.compare_tvm_with_tflite(pack_func, ifm_shapes, accel_type, enable_cascader=False)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,axis",
    [[(1, 2, 3, 4), 1], [(2, 3), 1], [(5, 6, 7), 2]],
)
def test_tflite_unpack(accel_type, ifm_shape, axis):
    np.random.seed(0)

    @tf.function
    def unpack_func(x):
        return tf.unstack(x, axis=axis)

    infra.compare_tvm_with_tflite(unpack_func, [ifm_shape], accel_type, enable_cascader=False)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape", [(1, 15, 15, 3), (1, 8, 9, 1)])
@pytest.mark.parametrize("alpha", [0.2, 0.634])
def test_tflite_leaky_relu(accel_type, ifm_shape, alpha):
    np.random.seed(0)

    @tf.function
    def leaky_relu_func(x):
        return tf.nn.leaky_relu(x, alpha=alpha)

    infra.compare_tvm_with_tflite(
        leaky_relu_func,
        [ifm_shape],
        accel_type,
        enable_cascader=is_u55_accel_type(accel_type),
        ranges=[(-1, 1)],
    )


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("ifm_shape", [(1, 14), (1, 151)])
@pytest.mark.parametrize("ofm_channels", [32, 64])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("activation_function", ["RELU", "NONE"])
def test_tflite_fully_connected(
    accel_type,
    ifm_shape,
    ofm_channels,
    use_bias,
    activation_function,
):
    np.random.seed(0)

    @tf.function
    def fully_connected(x):
        bias_shape = ofm_channels
        bias = tf.constant(np.random.uniform(size=bias_shape), dtype=tf.float32)
        w = tf.constant(
            np.random.uniform(size=[ifm_shape[1], ofm_channels]),
            dtype=tf.float32,
        )
        x = tf.matmul(x, w)
        if use_bias:
            x = tf.nn.bias_add(x, bias)
        if activation_function:
            x = tf.nn.relu(x)
        return x

    infra.compare_tvm_with_tflite(
        fully_connected, [ifm_shape], accel_type, enable_cascader=is_u55_accel_type(accel_type)
    )


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
