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
import tensorflow as tf
import tflite.Model

import tvm
from tvm.relay.backend.contrib.ethosu import legalize
from tvm import relay
from tvm.relay import dataflow_pattern
from tvm.relay.op.contrib import ethosu
from tvm.relay.backend.contrib.ethosu import util
from tests.python.contrib.test_ethosu.legalization import legalize_infra


@pytest.mark.parametrize(
    "ifm_shape, axis, keep_dims, use_same_quantization",
    [
        # mean to depthwise + multiply
        [(1, 8, 16, 16), (1, 2), True, False],
        [(1, 8, 16, 16), (2, 1), True, False],
        [(1, 3, 4), (0, 1), True, False],
        [(8, 5), (1, 0), True, False],
        [(1, 65, 2, 1), (1, 2), True, False],  # special case when h > 64
        # mean to average pool
        [(1, 8, 16, 16), (1,), True, True],
        [(1, 8, 16, 16), (2,), False, True],
        [(1, 8, 16, 16), (1, 2), False, True],
        [(3, 3, 4), (0,), True, True],
        [(3, 3, 4), (1,), False, True],
        [(8, 5), (0,), False, True],
        [(8, 5), (1,), True, True],
        # mean to depthwise
        [(1, 8, 16, 16), (1,), True, False],
        [(1, 8, 16, 16), (2,), True, False],
        [(1, 8, 16, 16), (1, 2), False, False],
        [(8, 4), (0,), False, False],
    ],
)
def test_mean(ifm_shape, axis, keep_dims, use_same_quantization):
    dtype = "int8"

    def create_tflite_graph():
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
        tflite_model = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model, 0)

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": ifm_shape},
            dtype_dict={"input": dtype},
        )
        return mod

    def create_relay_graph_with_same_quantization():
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
        return mod

    def verify(ext_func):
        out_var = ext_func.body

        next_op = out_var
        mul_op = None
        pooling_op = None
        depthwise_op = None
        if (
            isinstance(next_op, relay.expr.Call)
            and isinstance(next_op.op, tvm.ir.op.Op)
            and next_op.op.name == "reshape"
        ):
            next_op = next_op.args[0]
        if util.is_named_ethosu_op(next_op, "binary_elementwise"):
            mul_op = next_op
            next_op = next_op.args[0]
        if util.is_named_ethosu_op(next_op, "pooling"):
            pooling_op = next_op
            next_op = next_op.args[0]
        if util.is_named_ethosu_op(next_op, "depthwise_conv2d"):
            depthwise_op = next_op
            next_op = next_op.args[0]
        while (
            isinstance(next_op, relay.expr.Call)
            and isinstance(next_op.op, tvm.ir.op.Op)
            and next_op.op.name == "reshape"
        ):
            next_op = next_op.args[0]
        in_var = next_op

        def calculate_expected_output_shape():
            for i in range(len(ifm_shape)):
                if i in axis:
                    if keep_dims:
                        yield 1
                else:
                    yield ifm_shape[i]

        out_shape = tuple(calculate_expected_output_shape())

        # check IFM
        assert tuple(in_var.checked_type.shape) == ifm_shape
        assert in_var.checked_type.dtype == dtype

        # check OFM
        assert tuple(out_var.checked_type.shape) == out_shape
        assert out_var.checked_type.dtype == dtype

        # check expected legalization case
        if axis in [(1, 2), (2, 1), (0, 1), (1, 0)] and keep_dims and dtype == "int8":
            assert depthwise_op and mul_op
            assert mul_op.attrs.operator_type == "MUL"
        elif pooling_op:
            attrs = pooling_op.attrs
            assert (
                attrs.ifm_scale == attrs.ofm_scale and attrs.ifm_zero_point == attrs.ofm_zero_point
            )
        else:
            assert depthwise_op
            assert not mul_op

    rewriter = legalize.MeanRewriter()
    pattern_table = [
        (
            ethosu.MeanParams.composite_name,
            ethosu.mean_pattern(),
            lambda pat: ethosu.MeanParams(pat).is_valid(),
        ),
    ]

    mod = (
        create_relay_graph_with_same_quantization()
        if use_same_quantization
        else create_tflite_graph()
    )
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
