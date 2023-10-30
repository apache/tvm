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
"""Test that do-nothing requantize -> clip operators are removed during legalization."""

import numpy as np
import pytest

import tvm
from tvm import nd, relay
from tvm.relay import transform


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def tvm_const(obj):
    return relay.Constant(nd.array(obj))


@pytest.mark.parametrize(
    "dtype,min_val,max_val,is_redundant",
    [
        ("int8", -128, 127, True),
        ("int8", -127, 127, False),
        ("int16", -128, 127, False),
        ("int32", -2147483648, 2147483647, True),
    ],
)
def test_removes_redundant_requantize_clip_ops(dtype, min_val, max_val, is_redundant):
    """Test that qnn.requantize -> clip sequences are removed during legalization if the bounds of
    the clip operator match the min and max values of the data type."""

    input_var = relay.var("input", shape=(1, 3, 3, 4), dtype="int32")
    out = relay.qnn.requantize(
        input_var,
        tvm_const(np.float32(1.0)),
        tvm_const(np.int32(0)),
        tvm_const(np.float32(1.0)),
        tvm_const(np.int32(-128)),
        axis=3,
        out_dtype=dtype,
    )
    out = relay.clip(out, a_min=min_val, a_max=max_val)
    func = relay.Function([input_var], out)
    unmodified = run_opt_pass(func, transform.InferType())
    legalized = run_opt_pass(func, transform.Legalize())

    # Check that the clip op was removed if and only if `is_redundant` is True.
    if is_redundant:
        assert legalized.body.op.name == "qnn.requantize"
        assert not tvm.ir.structural_equal(unmodified, legalized)
    else:
        assert legalized.body.op.name == "clip"
        tvm.ir.assert_structural_equal(unmodified, legalized)


def test_ignores_standalone_clip_ops():
    """The legalization pass should only affect qnn.requantize -> clip sequences, and should leave
    standalone clip operators untouched."""

    input_var = relay.var("x", shape=(1, 3, 3, 4), dtype="int8")
    out = relay.clip(input_var, a_min=-128, a_max=127)
    func = relay.Function([input_var], out)
    unmodified = run_opt_pass(func, transform.InferType())
    legalized = run_opt_pass(func, transform.Legalize())
    tvm.ir.assert_structural_equal(unmodified, legalized)
