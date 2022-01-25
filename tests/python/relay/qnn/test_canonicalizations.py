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
from typing import Callable

import numpy as np
from tvm import relay
from tvm.relay.qnn.op import canonicalizations


class TestIntegerTableLookupTable:
    """Consists of tests testing functionality of creating lookup tables for integer operations."""

    def fake_identity_func_numpy(self, arr: np.ndarray):
        return arr.astype("float32")

    def fake_identity_func_relay(
        self,
        floating_point_func: Callable[[np.ndarray], np.ndarray],
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
                floating_point_func=floating_point_func,
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

    def run_function_test(
        self,
        in_scale: float,
        in_zero_point: int,
        out_scale: float,
        out_zero_point: int,
        in_dtype: str,
        out_dtype: str,
        floating_point_func: Callable[[np.ndarray], np.ndarray],
        input_arg: relay.Expr = None,
        rtol=1e-7,
        atol=0,
    ):
        relay_lookup, input_arg = self.fake_identity_func_relay(
            input_arg=input_arg,
            floating_point_func=floating_point_func,
            in_scale=relay.const(in_scale, "float32"),
            in_zero_point=relay.const(in_zero_point, "int32"),
            out_scale=relay.const(out_scale, "float32"),
            out_zero_point=relay.const(out_zero_point, "int32"),
            in_dtype=in_dtype,
            out_dtype=out_dtype,
        )
        result = canonicalizations.run_const_expr(relay_lookup)
        np.testing.assert_allclose(
            floating_point_func(
                self.dequantize_numpy(input_arg, np_scale=in_scale, np_zero_point=in_zero_point)
            ),
            self.dequantize_numpy(result, np_scale=out_scale, np_zero_point=out_zero_point),
            atol=atol,
            rtol=rtol,
        )

    """Test mapping between different input/output dtypes"""

    def test_int8_to_int8(self):
        self.run_function_test(
            in_scale=1.0,
            in_zero_point=0,
            out_scale=1.0,
            out_zero_point=0,
            in_dtype="int8",
            out_dtype="int8",
            floating_point_func=self.fake_identity_func_numpy,
        )

    def test_uint8_to_uint8(self):
        self.run_function_test(
            in_scale=1.0,
            in_zero_point=128,
            out_scale=1.0,
            out_zero_point=128,
            in_dtype="uint8",
            out_dtype="uint8",
            floating_point_func=self.fake_identity_func_numpy,
        )

    def test_int8_to_uint8(self):
        self.run_function_test(
            in_scale=1.0,
            in_zero_point=0,
            out_scale=1.0,
            out_zero_point=128,
            in_dtype="int8",
            out_dtype="uint8",
            floating_point_func=self.fake_identity_func_numpy,
        )

    def test_uint8_to_int8(self):
        self.run_function_test(
            in_scale=1.0,
            in_zero_point=128,
            out_scale=1.0,
            out_zero_point=0,
            in_dtype="uint8",
            out_dtype="int8",
            floating_point_func=self.fake_identity_func_numpy,
        )

    """Test different input shapes"""

    def test_keep_input_shapes(self):
        # input in floating point ~[-2, 2], final output ~[0, 8]
        self.run_function_test(
            input_arg=relay.const(np.arange(-128, 128).astype("int8").reshape([2, 2, 8, 8])),
            in_scale=0.015,
            in_zero_point=0,
            out_scale=16 / 256,
            out_zero_point=0,
            in_dtype="int8",
            out_dtype="int8",
            floating_point_func=self.fake_identity_func_numpy,
            atol=0.03,
            rtol=0.01,
        )
        self.run_function_test(
            input_arg=relay.const(np.arange(-128, 128).astype("int8").reshape([2, 2, 64])),
            in_scale=0.015,
            in_zero_point=0,
            out_scale=16 / 256,
            out_zero_point=0,
            in_dtype="int8",
            out_dtype="int8",
            floating_point_func=self.fake_identity_func_numpy,
            atol=0.03,
            rtol=0.01,
        )
        self.run_function_test(
            input_arg=relay.const(np.arange(-128, 128).astype("int8").reshape([2, 128])),
            in_scale=0.015,
            in_zero_point=0,
            out_scale=16 / 256,
            out_zero_point=0,
            in_dtype="int8",
            out_dtype="int8",
            floating_point_func=self.fake_identity_func_numpy,
            atol=0.03,
            rtol=0.01,
        )

    """Test mapping with different in/out qparams works."""

    def test_different_in_out_qparams(self):
        self.run_function_test(
            in_scale=1.0,
            in_zero_point=128,
            out_scale=1.0,
            out_zero_point=128,
            in_dtype="uint8",
            out_dtype="uint8",
            floating_point_func=self.fake_identity_func_numpy,
            atol=1,  # numbers range from -128 -> 128 so not that big error
            rtol=0,
        )

    """Test some simple functions"""

    def test_tanh(self):
        # 1 / 64 in scale -- input range is ~ (-2, 2), tanh(+-2) ~= +-1
        # 1 / 128 out_scale -- output range is ~(-1, 1)
        self.run_function_test(
            input_arg=relay.const(np.arange(-128, 128).astype("int8")),
            in_scale=1 / 64,
            in_zero_point=0,
            out_scale=1 / 128,
            out_zero_point=0,
            in_dtype="int8",
            out_dtype="int8",
            floating_point_func=np.tanh,
            atol=0.01,
            rtol=0.01,
        )

    def test_exp(self):
        # input in floating point ~[-2, 2], final output ~[0, 8]
        self.run_function_test(
            input_arg=relay.const(np.arange(-128, 128).astype("int8")),
            in_scale=0.015,
            in_zero_point=0,
            out_scale=16 / 256,
            out_zero_point=0,
            in_dtype="int8",
            out_dtype="int8",
            floating_point_func=np.exp,
            atol=0.03,
            rtol=0.01,
        )
