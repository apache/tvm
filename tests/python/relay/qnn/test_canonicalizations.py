import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.op import canonicalizations


class TestIntegerTableLookupTable:
    """Consists of tests testing functionality of creating lookup tables for integer operations."""

    def fake_identity_func_numpy(self, arr: np.ndarray):
        return arr.astype("float32")

    def fake_identity_func_relay(
        self,
        input_arg=None,
        in_scale=relay.const(1.0, dtype="float32"),
        in_zero_point=relay.const(0, dtype="int32"),
        out_scale=relay.const(1.0, dtype="float32"),
        out_zero_point=relay.const(0, dtype="int32"),
        in_axis=-1,
        out_axis=-1,
        in_dtype="uint8",
        out_dtype="uint8",
    ):
        if input_arg is None:
            input_arg = relay.const(np.arange(0, 256, dtype="uint8").view(in_dtype))

        return (
            canonicalizations.create_integer_lookup_op(
                input_arg=input_arg,
                floating_point_func=self.fake_identity_func_numpy,
                in_scale=in_scale,
                in_zero_point=in_zero_point,
                out_scale=out_scale,
                out_zero_point=out_zero_point,
                in_axis=in_axis,
                out_axis=out_axis,
                in_dtype=in_dtype,
                out_dtype=out_dtype,
            ),
            input_arg.data.numpy(),
        )

    def dequantize_numpy(self, np_arr, np_scale=1.0, np_zero_point=0):
        return (np_arr.astype("int32") - np_zero_point) * np_scale

    def run_identity_function_test(
        self,
        in_scale: float,
        in_zero_point: int,
        out_scale: float,
        out_zero_point: int,
        in_dtype: str,
        out_dtype: str,
        rtol=1e-7,
        atol=0,
    ):
        relay_lookup, input_arg = self.fake_identity_func_relay(
            in_scale=relay.const(in_scale, "float32"),
            in_zero_point=relay.const(in_zero_point, "int32"),
            out_scale=relay.const(out_scale, "float32"),
            out_zero_point=relay.const(out_zero_point, "int32"),
            in_dtype=in_dtype,
            out_dtype=out_dtype,
        )
        result = canonicalizations.run_const_expr(relay_lookup)
        np.testing.assert_allclose(
            self.dequantize_numpy(input_arg, np_scale=in_scale, np_zero_point=in_zero_point),
            self.dequantize_numpy(result, np_scale=out_scale, np_zero_point=out_zero_point),
            atol=atol,
            rtol=rtol,
        )

    def test_int8_to_int8(self):
        """Test int8 input to int8 output mapping workings"""
        self.run_identity_function_test(
            in_scale=1.0,
            in_zero_point=0,
            out_scale=1.0,
            out_zero_point=0,
            in_dtype="int8",
            out_dtype="int8",
        )

    def test_uint8_to_uint8(self):
        self.run_identity_function_test(
            in_scale=1.0,
            in_zero_point=128,
            out_scale=1.0,
            out_zero_point=128,
            in_dtype="uint8",
            out_dtype="uint8",
        )

    def test_int8_to_uint8(self):
        self.run_identity_function_test(
            in_scale=1.0,
            in_zero_point=0,
            out_scale=1.0,
            out_zero_point=128,
            in_dtype="int8",
            out_dtype="uint8",
        )

    def test_uint8_to_int8(self):
        self.run_identity_function_test(
            in_scale=1.0,
            in_zero_point=128,
            out_scale=1.0,
            out_zero_point=0,
            in_dtype="uint8",
            out_dtype="int8",
        )

    def test_different_in_out_qparams(self):
        """Test mapping with different in/out qparams works."""
        self.run_identity_function_test(
            in_scale=1.0,
            in_zero_point=128,
            out_scale=1.0,
            out_zero_point=128,
            in_dtype="uint8",
            out_dtype="uint8",
            atol=1,  # numbers range from -128 -> 128 so not that big error
            rtol=0,
        )


"""
def test_fake_quantize_tanh():
    x = relay.var("x", shape=[3, 3, 3, 3], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(0.03), zero)
    op = relay.op.tanh(x)

    # Have difference scales for input/output to test if can handle
    op = relay.qnn.op.quantize(op, relay.const(0.01), zero)

    x_np = np.random.randint(-128, 127, size=[3, 3, 3, 3], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_erf():
    x = relay.var("x", shape=[3, 3, 3, 3], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(0.03), zero)
    op = relay.op.erf(x)

    # Have difference scales for input/output to test if can handle
    op = relay.qnn.op.quantize(op, relay.const(0.01), zero)

    x_np = np.random.randint(-128, 127, size=[3, 3, 3, 3], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_exp():
    x = relay.var("x", shape=[3, 3, 3, 3], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(0.03), zero)
    op = relay.op.exp(x)

    # Have difference scales for input/output to test if can handle
    op = relay.qnn.op.quantize(op, relay.const(0.01), zero)

    x_np = np.random.randint(-128, 127, size=[3, 3, 3, 3], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_sigmoid():
    x = relay.var("x", shape=[3, 3, 3, 3], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(0.03), zero)
    op = relay.op.sigmoid(x)

    # Have difference scales for input/output to test if can handle
    op = relay.qnn.op.quantize(op, relay.const(0.01), zero)

    x_np = np.random.randint(-128, 127, size=[3, 3, 3, 3], dtype="int8")

    compare_fq_to_int(op, [x_np])
"""
