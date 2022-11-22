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

"""Unit tests for the convert equivalents pass."""

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from tvm.relay.op.contrib.ethosn import ConvertEquivalents
from tvm.relay import ExprVisitor

from . import infrastructure as tei
from .test_addition import _get_addition_qnn_params


def _assert_structural_equal(a, b):
    """Check structural equality of two Relay expressions."""
    reason = (
        "Actual and expected relay functions are not equal. "
        "ConvertEquivalents is not correctly transforming the input "
        "graph."
    )
    assert tvm.ir.structural_equal(a, b), reason


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape,channels", [((1, 4, 4, 8), 8), ((1, 16, 12, 4), 4)])
@pytest.mark.parametrize("reverse_inputs", [True, False])
def test_multiply_to_depthwise(dtype, shape, channels, reverse_inputs):
    """Check that multiply is correctly converted to a depthwise operation."""
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[3]
    )
    x = relay.var("x", shape=shape, dtype=dtype)
    constant_shape = (1, 1, 1, channels)
    y_data = np.random.randint(data_min, data_max + 1, size=constant_shape, dtype=dtype)

    def before():
        y = relay.const(y_data, dtype=dtype)
        expr = relay.qnn.op.mul(
            y if reverse_inputs else x,
            x if reverse_inputs else y,
            relay.const(input_sc, "float32"),
            relay.const(input_zp, "int32"),
            relay.const(input2_sc, "float32"),
            relay.const(input2_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_mul_to_depthwise")
        return tei.make_ethosn_partition(composite)

    def expected():
        constant_shape_hwoi = (1, 1, channels, 1)
        y_data_hwoi = y_data.reshape(constant_shape_hwoi)
        y_hwoi = relay.const(y_data_hwoi, dtype=dtype)
        expr = relay.qnn.op.conv2d(
            x,
            y_hwoi,
            relay.const(input2_zp if reverse_inputs else input_zp, "int32"),
            relay.const(input_zp if reverse_inputs else input2_zp, "int32"),
            relay.const(input2_sc if reverse_inputs else input_sc, "float32"),
            relay.const(input_sc if reverse_inputs else input2_sc, "float32"),
            (1, 1),
            channels,
            (1, 1),
            (0, 0),
            (1, 1),
            channels,
            "NHWC",
            "HWOI",
            "NHWC",
            "int32",
        )
        expr = relay.nn.bias_add(expr, relay.const(np.zeros((channels,), dtype="int32")), axis=3)
        expr = relay.qnn.op.requantize(
            expr,
            relay.const(input2_sc if reverse_inputs else input_sc, "float32"),
            relay.const(input2_zp if reverse_inputs else input_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
            out_dtype=dtype,
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_conv2d")
        return tei.make_ethosn_partition(composite)

    mod = before()
    mod = ConvertEquivalents()(mod)
    expected_mod = expected()
    _assert_structural_equal(mod["ethos-n_0"], expected_mod["ethos-n_0"])


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,shape,constant_shape",
    [("int8", (1, 4, 4), (4,)), ("int16", (1, 16, 12, 4), (1, 1, 1, 4))],
)
def test_unsupported_multiply_to_depthwise(dtype, shape, constant_shape):
    """Check that unsupported variants of multiply to depthwise are not converted."""
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[-1]
    )
    x = relay.var("x", shape=shape, dtype=dtype)
    y_data = np.random.randint(data_min, data_max + 1, size=constant_shape, dtype=dtype)

    def before():
        y = relay.const(y_data, dtype=dtype)
        expr = relay.qnn.op.mul(
            x,
            y,
            relay.const(input_sc, "float32"),
            relay.const(input_zp, "int32"),
            relay.const(input2_sc, "float32"),
            relay.const(input2_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_mul_to_depthwise")
        return tei.make_ethosn_partition(composite)

    mod = before()

    error_regex = (
        r'Operation "ethos-n.qnn_mul_to_depthwise" was marked '
        r"as having a valid conversion, but it could not be converted."
    )

    with pytest.raises(tvm.TVMError, match=error_regex):
        mod = ConvertEquivalents()(mod)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,constant_shape",
    [((1, 4, 4, 8), (1, 1, 1, 1)), ((1, 16, 12, 4), None)],
)
@pytest.mark.parametrize("reverse_inputs", [True, False])
def test_multiply_to_reinterpret_quantize(shape, constant_shape, reverse_inputs):
    """Check that multiply is correctly converted to a reinterpret quantize operation."""
    np.random.seed(0)

    dtype = "uint8"

    # Multiply can only be offloaded as a reinterpret quantize operation if
    # it is an identity option. We must choose the quantization and constant
    # data carefully to make sure that this is the case.
    input_zp = 0
    input_sc = 0.007814894430339336
    input2_zp = 0
    input2_sc = 0.5
    output_zp = 0
    output_sc = 0.9963990449905396
    constant_data = 255

    x = relay.var("x", shape=shape, dtype=dtype)
    y_data = np.array(constant_data, dtype=dtype).reshape(constant_shape)

    def before():
        y = relay.const(y_data, dtype=dtype)
        expr = relay.qnn.op.mul(
            y if reverse_inputs else x,
            x if reverse_inputs else y,
            relay.const(input2_sc if reverse_inputs else input_sc, "float32"),
            relay.const(input2_zp if reverse_inputs else input_zp, "int32"),
            relay.const(input_sc if reverse_inputs else input2_sc, "float32"),
            relay.const(input_zp if reverse_inputs else input2_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_mul_to_reinterpret_quantize")
        return tei.make_ethosn_partition(composite)

    def expected():
        expr = relay.qnn.op.requantize(
            x,
            relay.const(input_sc, "float32"),
            relay.const(input_zp if reverse_inputs else input_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
            out_dtype=dtype,
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_reinterpret_quantize")
        return tei.make_ethosn_partition(composite)

    mod = before()
    mod = ConvertEquivalents()(mod)
    expected_mod = expected()
    _assert_structural_equal(mod["ethos-n_0"], expected_mod["ethos-n_0"])


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,shape,constant_shape",
    [("float32", (1, 16, 12, 4), None)],
)
def test_unsupported_multiply_to_reinterpret_quantize(dtype, shape, constant_shape):
    """
    Check that unsupported variants of multiply conversion to reinterpret
    quantize are not converted.
    """
    np.random.seed(0)

    # Multiply can only be offloaded as a reinterpret quantize operation if
    # it is an identity option. We must choose the quantization and constant
    # data carefully to make sure that this is the case.
    input_zp = 0
    input_sc = 0.007814894430339336
    input2_zp = 0
    input2_sc = 0.5
    output_zp = 0
    output_sc = 0.9963990449905396
    constant_data = 255

    x = relay.var("x", shape=shape, dtype=dtype)
    y_data = np.array(constant_data, dtype=dtype).reshape(constant_shape)

    def before():
        y = relay.const(y_data, dtype=dtype)
        expr = relay.qnn.op.mul(
            x,
            y,
            relay.const(input_sc, "float32"),
            relay.const(input_zp, "int32"),
            relay.const(input2_sc, "float32"),
            relay.const(input2_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_mul_to_reinterpret_quantize")
        return tei.make_ethosn_partition(composite)

    mod = before()

    error_regex = (
        r'Operation "ethos-n.qnn_mul_to_reinterpret_quantize" was marked '
        r"as having a valid conversion, but it could not be converted."
    )

    with pytest.raises(tvm.TVMError, match=error_regex):
        mod = ConvertEquivalents()(mod)


@requires_ethosn
@pytest.mark.parametrize("reverse_inputs", [True, False])
def test_add_to_depthwise(reverse_inputs):
    """
    Check that add is converted correctly.
    """
    dtype = "uint8"
    lhs_shape = (1, 2, 4, 8)
    rhs_shape = (1, 1, 1, 8)
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    x = relay.var("x", shape=lhs_shape, dtype=dtype)
    y_data = np.random.randint(data_min, data_max + 1, size=rhs_shape, dtype=dtype)

    def before():
        y = relay.const(y_data)
        expr = relay.qnn.op.add(
            lhs=y if reverse_inputs else x,
            rhs=x if reverse_inputs else y,
            lhs_scale=relay.const(lhs_sc, "float32"),
            lhs_zero_point=relay.const(lhs_zp, "int32"),
            rhs_scale=relay.const(rhs_sc, "float32"),
            rhs_zero_point=relay.const(rhs_zp, "int32"),
            output_scale=relay.const(out_sc, "float32"),
            output_zero_point=relay.const(out_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_add_to_depthwise")
        return tei.make_ethosn_partition(composite)

    class ConversionChecker(ExprVisitor):
        """
        Pass to check the new composite function is in the expected format.
        """

        sequence = ["qnn.conv2d", "nn.bias_add", "qnn.requantize"]

        # pylint: disable=invalid-name
        def visit_function(self, fn):
            composite_name = fn.attrs["Composite"]
            expected = "ethos-n.qnn_conv2d"
            assert (
                composite_name == expected
            ), f"Expected Composite attribute {expected} but got {composite_name}"
            super().visit_function(fn)

        def visit_call(self, call):
            op_name = call.op.name
            expected_name = self.sequence.pop()
            assert op_name == expected_name, f"Got operator {op_name} but expected {expected_name}"
            super().visit_call(call)

    mod = before()
    mod = ConvertEquivalents()(mod)
    mod = ConversionChecker().visit(mod["ethos-n_0"].body.op)


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,lhs_shape,rhs_shape", [("uint8", (1, 4, 4), (1, 1, 4)), ("int16", (1, 4, 4, 4), (4,))]
)
def test_unsupported_add_to_depthwise(dtype, lhs_shape, rhs_shape):
    """Check that unsupported variants of add are not converted."""
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    x = relay.var("x", shape=lhs_shape, dtype=dtype)
    y_data = np.random.randint(data_min, data_max + 1, size=rhs_shape, dtype=dtype)

    def before():
        y = relay.const(y_data)
        expr = relay.qnn.op.add(
            lhs=x,
            rhs=y,
            lhs_scale=relay.const(lhs_sc, "float32"),
            lhs_zero_point=relay.const(lhs_zp, "int32"),
            rhs_scale=relay.const(rhs_sc, "float32"),
            rhs_zero_point=relay.const(rhs_zp, "int32"),
            output_scale=relay.const(out_sc, "float32"),
            output_zero_point=relay.const(out_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_add_to_depthwise")
        return tei.make_ethosn_partition(composite)

    mod = before()

    error_regex = (
        r'Operation "ethos-n.qnn_add_to_depthwise" was marked '
        r"as having a valid conversion, but it could not be converted."
    )

    with pytest.raises(tvm.TVMError, match=error_regex):
        mod = ConvertEquivalents()(mod)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,constant_shape",
    [
        ((1, 4, 4, 8), (1, 1, 1, 1)),
        ((1, 16, 12, 4), None),
    ],
)
@pytest.mark.parametrize("reverse_inputs", [True, False])
def test_add_to_reinterpret_quantize(shape, constant_shape, reverse_inputs):
    """Check that add is correctly converted to a reinterpret quantize operation."""
    np.random.seed(0)

    dtype = "uint8"

    # Add can only be offloaded as a reinterpret quantize operation if
    # it is an identity option. We must choose the quantization and constant
    # data carefully to make sure that this is the case.
    input_zp = 128
    input_sc = 0.0078125
    input2_zp = 0
    input2_sc = 0.003921568859368563
    output_zp = 0
    output_sc = 0.007814894430339336
    constant_data = 255

    x = relay.var("x", shape=shape, dtype=dtype)
    y_data = np.array(constant_data, dtype=dtype).reshape(constant_shape)

    def before():
        y = relay.const(y_data, dtype=dtype)
        expr = relay.qnn.op.add(
            y if reverse_inputs else x,
            x if reverse_inputs else y,
            relay.const(input2_sc if reverse_inputs else input_sc, "float32"),
            relay.const(input2_zp if reverse_inputs else input_zp, "int32"),
            relay.const(input_sc if reverse_inputs else input2_sc, "float32"),
            relay.const(input_zp if reverse_inputs else input2_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_add_to_reinterpret_quantize")
        return tei.make_ethosn_partition(composite)

    def expected():
        expr = relay.qnn.op.requantize(
            x,
            relay.const(input_sc, "float32"),
            relay.const(input_zp if reverse_inputs else input_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
            out_dtype=dtype,
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_reinterpret_quantize")
        return tei.make_ethosn_partition(composite)

    mod = before()
    mod = ConvertEquivalents()(mod)
    expected_mod = expected()
    _assert_structural_equal(mod["ethos-n_0"], expected_mod["ethos-n_0"])


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,shape,constant_shape",
    [
        ("float32", (1, 16, 12, 4), None),
    ],
)
def test_unsupported_add_to_reinterpret_quantize(dtype, shape, constant_shape):
    """Check that unsupported variants of add to reinterpret quantize are not converted."""
    np.random.seed(0)

    # Add can only be offloaded as a reinterpret quantize operation if
    # it is an identity option. We must choose the quantization and constant
    # data carefully to make sure that this is the case.
    input_zp = 128
    input_sc = 0.0078125
    input2_zp = 0
    input2_sc = 0.003921568859368563
    output_zp = 0
    output_sc = 0.007814894430339336
    constant_data = 255

    x = relay.var("x", shape=shape, dtype=dtype)
    y_data = np.array(constant_data, dtype=dtype).reshape(constant_shape)

    def before():
        y = relay.const(y_data, dtype=dtype)
        expr = relay.qnn.op.add(
            x,
            y,
            relay.const(input_sc, "float32"),
            relay.const(input_zp, "int32"),
            relay.const(input2_sc, "float32"),
            relay.const(input2_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
        )
        composite = tei.make_ethosn_composite(expr, "ethos-n.qnn_add_to_reinterpret_quantize")
        return tei.make_ethosn_partition(composite)

    mod = before()

    error_regex = (
        r'Operation "ethos-n.qnn_add_to_reinterpret_quantize" was marked '
        r"as having a valid conversion, but it could not be converted."
    )

    with pytest.raises(tvm.TVMError, match=error_regex):
        mod = ConvertEquivalents()(mod)
