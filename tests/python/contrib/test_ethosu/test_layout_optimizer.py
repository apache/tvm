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

"""Test the layout optimization pass. This pass is used to
convert subgraphs to the preferred layout of NHCWB16.
"""

import pytest

pytest.importorskip("ethosu.vela")

import sys

import numpy as np
import tensorflow as tf
import tflite.Model

import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tvm.relay.backend.contrib.ethosu.codegen import LayoutOptimizer
from tvm.relay.backend.contrib.ethosu.codegen import relay_to_tir

from . import infra


def _optimize(func, optimize=True):
    """Create IRModule and run layout optimizer pass."""
    func = func.with_attr("Compiler", "ethos-u")
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    if optimize:
        mod = LayoutOptimizer()(mod)
    entry = mod["main"]
    return entry if isinstance(func, relay.Function) else entry.body


def _assert_structural_equal(a, b):
    """Check structural equality of two Relay expressions."""
    reason = (
        "Actual and expected relay functions are not equal. "
        "LayoutOptimizer is not correctly converting layouts."
    )
    assert tvm.ir.structural_equal(a, b), reason


def _compile_and_compare_model(tflite_graph, ifm_shape, dtype):
    """Compare running result of compilation against TFLite."""
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={
            "ifm": ifm_shape,
        },
        dtype_dict={
            "ifm": dtype,
        },
    )
    mod = partition_for_ethosu(mod, params)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

    test_runner = infra.create_test_runner("ethos-u55-256")
    compiled_models = infra.build_source(
        mod,
        input_data,
        output_data,
        test_runner,
        output_tolerance=0,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    get_artifacts = tvm._ffi.get_global_func("runtime.module.ethos-u.get_artifacts")
    compilation_artifacts = get_artifacts(ethosu_module)
    cmms = bytes.fromhex(compilation_artifacts[0].command_stream)
    infra.print_payload(cmms)
    infra.verify_source(compiled_models, test_runner)


def test_single_convolution():
    """Test a single convolution to make sure the layouts remain
    unaltered.
    """

    def get_graph():
        x = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
        x = infra.make_ethosu_conv2d(
            ifm=x,
            ifm_channels=8,
            ofm_channels=8,
            kernel_shape=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(x), x)

    a = _optimize(get_graph())
    b = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(a, b)


def test_multiple_convolution():
    """Test layout optimization pass on linear chain of convolutions. I.e,

    conv_1
      |
    conv_2
      |
    conv_3
    """

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
        for i in range(3):
            ifm_layout = "NHCWB16" if get_expected and i != 0 else "NHWC"
            ofm_layout = "NHCWB16" if get_expected and i != 2 else "NHWC"
            x = infra.make_ethosu_conv2d(
                ifm=x,
                ifm_channels=8,
                ofm_channels=8,
                kernel_shape=(1, 1),
                padding=(0, 0),
                strides=(1, 1),
                dilation=(1, 1),
                ifm_layout=ifm_layout,
                ofm_layout=ofm_layout,
            )
        return relay.Function(relay.analysis.free_vars(x), x)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_multiple_depthwise_convolution():
    """Test layout optimization pass on multiple depthwise convolutions.

    depthwise_conv_1
           |
    depthwise_conv_2
           |
    depthwise_conv_3
    """

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
        for i in range(3):
            ifm_layout = "NHCWB16" if get_expected and i != 0 else "NHWC"
            ofm_layout = "NHCWB16" if get_expected and i != 2 else "NHWC"
            x = infra.make_ethosu_depthwise_conv2d(
                ifm=x,
                channels=4,
                kernel_shape=(1, 1),
                padding=(0, 0),
                strides=(1, 1),
                dilation=(1, 1),
                ifm_layout=ifm_layout,
                ofm_layout=ofm_layout,
            )
        return relay.Function(relay.analysis.free_vars(x), x)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_ignore_transform_operations():
    """Test layout optimization pass ignores transform operations
    such as reshape and strided slice.

       conv_1
         |
      reshape
         |
    strided_slice
         |
       conv_2
    """

    def get_graph():
        in_1 = relay.var("x", shape=(1, 16, 16, 8), dtype="int8")
        conv_1 = infra.make_ethosu_conv2d(
            ifm=in_1,
            ifm_channels=8,
            ofm_channels=8,
            kernel_shape=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        reshape = relay.reshape(conv_1, (1, 16, 16, 8))
        strided_slice = relay.strided_slice(reshape, (0, 0, 0, 0), (1, 16, 16, 8))
        conv_2 = infra.make_ethosu_conv2d(
            ifm=strided_slice,
            ifm_channels=8,
            ofm_channels=8,
            kernel_shape=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(conv_2), conv_2)

    a = _optimize(get_graph())
    b = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(a, b)


def test_ignore_concatenate():
    """Test layout optimization pass ignores the concatenate operation,
    when layout transformation cannot occur.

    in_1     in_2
      \       /
       \   conv_1
        \   /
       concat
         |
       conv_2
    """

    def get_graph():
        in_1 = relay.var("x", shape=(1, 16, 16, 8), dtype="int8")
        in_2 = relay.var("y", shape=(1, 16, 16, 8), dtype="int8")
        conv_1 = infra.make_ethosu_conv2d(
            ifm=in_2,
            ifm_channels=8,
            ofm_channels=8,
            kernel_shape=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        concat = relay.concatenate([in_1, conv_1], axis=1)
        conv_2 = infra.make_ethosu_conv2d(
            ifm=concat,
            ifm_channels=8,
            ofm_channels=4,
            kernel_shape=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(conv_2), conv_2)

    a = _optimize(get_graph())
    b = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(a, b)


def test_ignore_concatnate_with_layout_transform():
    """Test the layout optimization pass ignores the concatenate
    operation and performs a layout transformation.

     in_1       in_2
      \          /
     pool_1   pool_2
        \      /
         concat
           |
         pool_3
    """

    def get_graph():
        in_1 = relay.var("x", shape=(1, 16, 16, 8), dtype="int8")
        in_2 = relay.var("y", shape=(1, 16, 16, 8), dtype="int8")
        pool_1 = infra.make_ethosu_pooling(
            in_1,
            "MAX",
            (1, 1),
            ofm_channels=8,
            strides=(1, 1),
            padding=(0, 0),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        pool_2 = infra.make_ethosu_pooling(
            in_2,
            "MAX",
            (1, 1),
            ofm_channels=8,
            strides=(1, 1),
            padding=(0, 0),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        concat = relay.concatenate([pool_1, pool_2], axis=1)
        pool_3 = infra.make_ethosu_pooling(
            concat,
            "MAX",
            (1, 1),
            ofm_channels=8,
            strides=(1, 1),
            padding=(0, 0),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(pool_3), pool_3)

    a = _optimize(get_graph())
    b = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(a, b)


def test_multiple_inputs():
    """Test the layout optimization pass works as expected when there
    are multiple inputs in the graph.

    pool_1 pool_2 pool_3
      \     |      /
       \    |    /
         concat
           |
         conv
    """

    def get_graph():
        poolings = []
        for _ in range(3):
            inp = relay.var("x", shape=(1, 3, 3, 4), dtype="int8")
            pool = infra.make_ethosu_pooling(
                inp,
                "MAX",
                (1, 1),
                ofm_channels=4,
                strides=(1, 1),
                padding=(0, 0),
                ifm_layout="NHWC",
                ofm_layout="NHWC",
            )
            poolings.append(pool)
        concat = relay.concatenate(poolings, axis=0)
        conv = infra.make_ethosu_conv2d(
            ifm=concat,
            ifm_channels=8,
            ofm_channels=4,
            kernel_shape=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            ifm_layout="NHWC",
            ofm_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(conv), conv)

    a = _optimize(get_graph())
    b = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(a, b)


def test_multiple_outputs():
    """Test the layout optimization pass works as expected when there
    are multiple outputs in the graph.

          pool_1
       /    |   \
  pool_2 pool_3 pool_4
        \   |   /
         concat
    """

    def get_graph(get_expected=False):
        in_1 = relay.var("x", shape=(1, 4, 4, 8), dtype="int8")
        pool_1 = infra.make_ethosu_pooling(
            in_1,
            "MAX",
            (1, 1),
            ofm_channels=4,
            strides=(1, 1),
            padding=(0, 0),
            ifm_layout="NHWC",
            ofm_layout="NHCWB16" if get_expected else "NHWC",
        )
        poolings = []
        for _ in range(3):
            poolings.append(
                infra.make_ethosu_pooling(
                    pool_1,
                    "MAX",
                    (1, 1),
                    ofm_channels=4,
                    strides=(1, 1),
                    padding=(0, 0),
                    ifm_layout="NHCWB16" if get_expected else "NHWC",
                    ofm_layout="NHWC",
                )
            )
        concat = relay.concatenate(poolings, axis=0)
        return relay.Function(relay.analysis.free_vars(concat), concat)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_multiple_binary_elementwise():
    """Test the layout optimization pass works as expected for
    binary elementwise operations.

    add_1  add_2
      \     /
       \   /
       add_3
    """

    def get_graph(get_expected=False):
        in_1 = relay.var("x", shape=(1, 2, 2, 2), dtype="int8")
        in_2 = relay.var("y", shape=(1, 2, 2, 2), dtype="int8")
        in_3 = relay.var("z", shape=(1, 2, 2, 2), dtype="int8")
        add_1 = infra.make_ethosu_binary_elementwise(
            in_1,
            in_2,
            ifm_channels=2,
            ifm2_channels=2,
            operator_type="ADD",
            ofm_dtype="int8",
            ifm_layout="NHWC",
            ifm2_layout="NHWC",
            ofm_layout="NHCWB16" if get_expected else "NHWC",
        )
        add_2 = infra.make_ethosu_binary_elementwise(
            in_2,
            in_3,
            ifm_channels=2,
            ifm2_channels=2,
            operator_type="ADD",
            ofm_dtype="int8",
            ifm_layout="NHWC",
            ifm2_layout="NHWC",
            ofm_layout="NHCWB16" if get_expected else "NHWC",
        )
        add_3 = infra.make_ethosu_binary_elementwise(
            add_1,
            add_2,
            ifm_channels=2,
            ifm2_channels=2,
            operator_type="ADD",
            ofm_dtype="int8",
            ifm_layout="NHCWB16" if get_expected else "NHWC",
            ifm2_layout="NHCWB16" if get_expected else "NHWC",
            ofm_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(add_3), add_3)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_multiple_pooling():
    """Test the layout optimization pass works as expected for
    multiple pooling operations.

    pool_1
      |
    pool_2
      |
    pool_3
    """

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
        for i in range(3):
            ifm_layout = "NHCWB16" if get_expected and i != 0 else "NHWC"
            ofm_layout = "NHCWB16" if get_expected and i != 2 else "NHWC"
            x = infra.make_ethosu_pooling(
                x,
                "MAX",
                (1, 1),
                ofm_channels=4,
                strides=(1, 1),
                padding=(0, 0),
                ifm_layout=ifm_layout,
                ofm_layout=ofm_layout,
            )
        return relay.Function(relay.analysis.free_vars(x), x)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_multiple_unary_elementwise():
    """Test the layout optimization pass works as expected for multiple
    unary elementwise operations.

    abs_1
      |
    abs_2
      |
    abs_3
    """

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
        for i in range(3):
            ifm_layout = "NHCWB16" if get_expected and i != 0 else "NHWC"
            ofm_layout = "NHCWB16" if get_expected and i != 2 else "NHWC"
            x = infra.make_ethosu_unary_elementwise(
                x,
                ofm_channels=4,
                operator_type="ABS",
                ifm_layout=ifm_layout,
                ofm_layout=ofm_layout,
            )
        return relay.Function(relay.analysis.free_vars(x), x)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_op_without_ethosu_consumer():
    """Test the layout optimization pass works as expected when
    there is a case that the output layout should not be altered
    since not all consumers are NPU operations (in this case conv).

    depthwise
        |
      conv
      /  \
     |  pool
     \   /
    (concat)
    """

    def get_graph(get_expected=False):
        exp_layout = "NHCWB16" if get_expected else "NHWC"

        x = relay.var("x", shape=(1, 2, 2, 2), dtype="int8")
        depthwise = infra.make_ethosu_depthwise_conv2d(
            x, 2, (1, 1), (0, 0), (1, 1), (0, 0), ofm_layout=exp_layout
        )
        conv = infra.make_ethosu_conv2d(
            depthwise,
            2,
            2,
            (1, 1),
            (0, 0),
            (1, 1),
            (0, 0),
            ifm_layout=exp_layout,
        )
        pool = infra.make_ethosu_pooling(conv, "MAX", (1, 1), 2, (1, 1), (0, 0))
        concat = relay.concatenate([conv, pool], axis=0)
        return relay.Function(relay.analysis.free_vars(concat), concat)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_diamond_graph():
    """
    Test the layout optimizer pass works as expected on a diamond graph
    with a case where the operation dominating the output operation
    cannot be altered, but operations within the diamond can.

      pool_1
        |
      pool_2
      /   \
     |  pool_3
     |     |
     |  pool_4
     |     |
     |  pool_5
     \    /
    (concat)
    """

    def get_graph(get_expected=False):
        exp_layout = "NHCWB16" if get_expected else "NHWC"
        x = relay.var("x", shape=(1, 2, 2, 2), dtype="int8")
        pool_1 = infra.make_ethosu_pooling(
            x, "MAX", (1, 1), 2, (1, 1), (0, 0), ofm_layout=exp_layout
        )
        pool_2 = infra.make_ethosu_pooling(
            pool_1, "MAX", (1, 1), 2, (1, 1), (0, 0), ifm_layout=exp_layout
        )
        pool_3 = infra.make_ethosu_pooling(
            pool_2, "MAX", (1, 1), 2, (1, 1), (0, 0), ofm_layout=exp_layout
        )
        pool_4 = infra.make_ethosu_pooling(
            pool_3, "MAX", (1, 1), 2, (1, 1), (0, 0), ifm_layout=exp_layout, ofm_layout=exp_layout
        )
        pool_5 = infra.make_ethosu_pooling(
            pool_4, "MAX", (1, 1), 2, (1, 1), (0, 0), ifm_layout=exp_layout
        )
        concat = relay.concatenate([pool_2, pool_5], axis=0)
        return relay.Function(relay.analysis.free_vars(concat), concat)

    a = _optimize(get_graph())
    b = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(a, b)


def test_same_output_multiple_convolutions():
    """Test running the layout optimization pass with multiple convolutions
    gives same output as TFLite."""

    np.random.seed(0)
    dtype = "int8"
    ifm_shape = (1, 8, 8, 32)
    kernel_shape = (1, 1, 32, 32)

    def create_model():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                for _ in range(3):
                    x = tf.nn.conv2d(
                        x,
                        filters=tf.constant(np.random.uniform(size=kernel_shape), dtype=tf.float32),
                        strides=(1, 1),
                        padding="SAME",
                        data_format="NHWC",
                        dilations=1,
                    )
                return x

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
        return converter.convert()

    _compile_and_compare_model(create_model(), ifm_shape, dtype)


def test_same_output_multiple_pooling():
    """Test running the layout optimization pass with multiple pooling
    operations gives same output as TFLite."""

    np.random.seed(0)
    dtype = "int8"
    ifm_shape = (1, 4, 2, 7)

    def create_model():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                for _ in range(2):
                    x = tf.nn.max_pool2d(x, (1, 1), (1, 1), "SAME", "NHWC")
                return x

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
        return converter.convert()

    _compile_and_compare_model(create_model(), ifm_shape, dtype)


def test_layout_optimizer_runs_in_compilation_pipeline():
    """Checks that the layout optimization pass runs as part of the NPU compilation
    pipeline."""

    def get_graph():
        x = relay.var("x", shape=(1, 4, 4, 4), dtype="int8")
        for _ in range(2):
            x = relay.nn.max_pool2d(x, layout="NHWC")

        func = relay.Function(relay.analysis.free_vars(x), x)
        return tvm.IRModule.from_expr(func)

    mod = get_graph()
    mod = partition_for_ethosu(mod)
    mod = relay_to_tir(mod)

    external_gv_name = mod["main"].body.op.name_hint
    prim_func = mod[external_gv_name]

    # Check for hints in the TIR prim func that the layout optimization pass has ran
    ops = prim_func.body.body.seq
    max_pool1, max_pool2 = ops

    assert str(max_pool1.value.args[31]) == '"NHCWB16"'
    assert str(max_pool2.value.args[14]) == '"NHCWB16"'


if __name__ == "__main__":
    pytest.main([__file__] + sys.argv[1:])
