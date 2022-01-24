import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.op import canonicalizations


class TestIntegerTableLookupTable:
    """Consists of tests testing functionality of creating lookup tables for integer operations."""

    # def __init__(self) -> None:
    #     self.input = np.arange(start=0, stop=256, dtype="uint8")

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

    def test_int8_to_int8(self):
        relay_lookup, input_arg = self.fake_identity_func_relay(in_dtype="int8", out_dtype="int8")
        result = canonicalizations.run_const_expr(relay_lookup)
        assert np.allclose(self.dequantize_numpy(input_arg), self.dequantize_numpy(result))

    def test_uint8_to_uint8(self):
        relay_lookup, input_arg = self.fake_identity_func_relay(in_dtype="uint8", out_dtype="uint8")
        result = canonicalizations.run_const_expr(relay_lookup)
        assert np.allclose(self.dequantize_numpy(input_arg), self.dequantize_numpy(result))

    def test_int8_to_uint8(self):
        relay_lookup, input_arg = self.fake_identity_func_relay(
            out_scale=relay.const(1.0, dtype="float32"),
            out_zero_point=relay.const(128, dtype="int32"),
            in_dtype="int8",
            out_dtype="uint8",
        )
        result = canonicalizations.run_const_expr(relay_lookup)
        assert np.allclose(
            self.dequantize_numpy(input_arg),
            self.dequantize_numpy(result, np_scale=1.0, np_zero_point=128),
        )

    def test_uint8_to_int8(self):
        relay_lookup, input_arg = self.fake_identity_func_relay(
            in_scale=relay.const(1.0, dtype="float32"),
            in_zero_point=relay.const(128, dtype="int32"),
            in_dtype="uint8",
            out_dtype="int8",
        )
        result = canonicalizations.run_const_expr(relay_lookup)
        assert np.allclose(
            self.dequantize_numpy(input_arg, np_scale=1.0, np_zero_point=128),
            self.dequantize_numpy(result),
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
