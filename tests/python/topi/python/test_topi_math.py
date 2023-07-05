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

import sys

import numpy as np
import pytest
import scipy
from scipy import special

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import te, topi
from tvm.topi import utils


def test_util():
    x = tvm.tir.const(100, "int32")
    assert utils.get_const_int(x) == 100
    assert utils.get_const_tuple((x, x)) == (100, 100)


ewise_operations = {
    "floor": {"topi": topi.floor, "ref": np.floor, "input_range": (-100, 100)},
    "ceil": {"topi": topi.ceil, "ref": np.ceil, "input_range": (-100, 100)},
    "sign": {
        "topi": topi.sign,
        "ref": np.sign,
        "input_range": (-100, 100),
        "skip_name_check": True,
    },
    "trunc": {"topi": topi.trunc, "ref": np.trunc, "input_range": (-100, 100)},
    "fabs": {"topi": topi.abs, "ref": np.fabs, "input_range": (-100, 100)},
    "round": {"topi": topi.round, "ref": np.round, "input_range": (-100, 100), "check_round": True},
    "exp": {"topi": topi.exp, "ref": np.exp, "input_range": (-1, 1)},
    "tanh": {
        "topi": topi.tanh,
        "ref": np.tanh,
        "input_range": (-10, 10),
        "shape": (128, 128),
        "dtype": ["float32", "float64"],
    },
    "sigmoid": {
        "topi": topi.sigmoid,
        "ref": lambda x: 1 / (1 + np.exp(-x)),
        "input_range": (-1, 1),
    },
    "log": {"topi": topi.log, "ref": np.log, "input_range": (0, 100)},
    "sqrt": {"topi": topi.sqrt, "ref": np.sqrt, "input_range": (0, 100)},
    "rsqrt": {
        "topi": topi.rsqrt,
        "ref": lambda x: np.ones_like(x) / np.sqrt(x),
        "input_range": (0, 100),
        "skip_name_check": True,
    },
    "cos": {"topi": topi.cos, "ref": np.cos, "input_range": (-2.0 * np.pi, 2.0 * np.pi)},
    "tan": {
        "topi": topi.tan,
        "ref": np.tan,
        "input_range": (-2.0 * np.pi, 2.0 * np.pi),
        "dtypes": ["float32", "float64"],
    },
    "sin": {"topi": topi.sin, "ref": np.sin, "input_range": (-2.0 * np.pi, 2.0 * np.pi)},
    "erf": {"topi": topi.erf, "ref": scipy.special.erf, "input_range": (-0.1, 0.1)},
    "isnan": {
        "topi": topi.isnan,
        "ref": np.isnan,
        "input_range": (-1, 1),
        "replace_with_nan": True,
    },
    "isfinite": {
        "topi": topi.isfinite,
        "ref": np.isfinite,
        "input_range": (0, 1),
        "shape": (8, 8),
        "skip_name_check": True,
        "replace_with_nan": True,
        "replace_with_inf": True,
        "dtypes": ["float32", "float64", "int32", "int16"],
    },
    "isinf": {
        "topi": topi.isinf,
        "ref": np.isinf,
        "input_range": (0, 1),
        "shape": (8, 8),
        "skip_name_check": True,
        "replace_with_nan": True,
        "replace_with_inf": True,
        "dtypes": ["float32", "float64", "int32", "int16"],
    },
    "fast_exp": {
        "topi": topi.fast_exp,
        "ref": np.exp,
        "skip_name_check": True,
        "input_range": (-88, 88),
        "step": 0.01,
    },
    "fast_erf": {
        "topi": topi.fast_erf,
        "ref": scipy.special.erf,
        "skip_name_check": True,
        "input_range": (-10, 10),
        "step": 0.01,
        "dtypes": ["float32", "float16"],
        "cast_output": True,
        "tolerance": [1e-5, 1e-1],
    },
    "fast_tanh": {
        "topi": topi.fast_tanh,
        "ref": np.tanh,
        "skip_name_check": True,
        "input_range": (-10, 10),
        "step": 0.01,
    },
}

topi_name, dtype, tolerance = tvm.testing.parameters(
    *[
        (name, dtype, config.get("tolerance", [1e-5] * len(dtype))[i])
        for name, config in ewise_operations.items()
        for i, dtype in enumerate(config.get("dtypes", ["float32"]))
    ]
)


@tvm.testing.fixture(cache_return_value=True)
def ewise_ref_data(topi_name, dtype):
    config = ewise_operations[topi_name]

    input_range = config["input_range"]
    shape = config.get("shape", (20, 3))

    a_np = np.random.uniform(*input_range, size=shape).astype(dtype)

    if dtype.startswith("float"):
        if config.get("replace_with_nan", False):
            a_np.ravel()[np.random.choice(a_np.size, int(a_np.size * 0.5), replace=False)] = np.nan
        if config.get("replace_with_inf", False):
            a_np.ravel()[
                np.random.choice(a_np.size, int(a_np.size * 0.5), replace=False)
            ] = np.infty

    # avoid round check too close to boundary
    if topi_name == "round":
        a_np += ((np.abs(np.fmod(a_np, 1)) - 0.5) < 1e-6) * 1e-4

    b_np = config["ref"](a_np)

    if config.get("cast_output", False):
        b_np = b_np.astype(dtype)

    return a_np, b_np


def test_ewise(target, dev, topi_name, dtype, tolerance, ewise_ref_data):
    target = tvm.target.Target(target)
    if target.kind.name == "vulkan" and topi_name in ["tan", "erf", "isnan", "isfinite", "isinf"]:
        pytest.xfail(f"Vulkan runtime doesn't support {topi_name} yet")

    topi_op = ewise_operations[topi_name]["topi"]
    skip_name_check = ewise_operations[topi_name].get("skip_name_check", False)

    m = te.var("m")
    l = te.var("l")
    A = te.placeholder((m, l), dtype=dtype, name="A")

    B = topi_op(A)
    assert tuple(B.shape) == tuple(A.shape)
    if not skip_name_check:
        assert B.op.body[0].op.name == "tir." + topi_name

    a_np, b_np = ewise_ref_data

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_injective_schedule(target)(B)
    foo = tvm.build(s, [A, B], target, name=topi_name)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros_like(b_np), dev)
    foo(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=tolerance, atol=tolerance)


from_dtype, to_dtype = tvm.testing.parameters(
    ("int32", "float32"),
    ("int32", "float64"),
    ("int32", "bool"),
    ("float16", "float32"),
    ("float16", "float64"),
    ("float32", "int32"),
    ("float32", "float64"),
    ("float32", "bool"),
    # disable this due to llvm5+ bug https://github.com/llvm/llvm-project/issues/56204
    # TODO (yongwww): pattern match f64->f16 to f64->f32->f16 as a workaround
    # ("float64", "float16"),
    ("float64", "float32"),
    ("bool", "float32"),
    ("bool", "int32"),
)


@tvm.testing.fixture(cache_return_value=True)
def cast_ref_data(from_dtype, to_dtype):
    shape = (5, 4)
    input_range = (-100, 100)

    if from_dtype == "bool":
        a_np = np.random.choice([True, False], size=shape)
    else:
        a_np = np.random.uniform(*input_range, size=shape).astype(from_dtype)

    if to_dtype == "bool":
        a_np = a_np - a_np[2, 3]
    b_np = a_np.astype(to_dtype)

    return a_np, b_np


def test_cast(target, dev, cast_ref_data, from_dtype, to_dtype):
    m = te.var("m")
    l = te.var("l")
    A = te.placeholder((m, l), dtype=from_dtype, name="A")
    B = topi.cast(A, to_dtype)

    a_np, b_np = cast_ref_data

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_injective_schedule(target)(B)
    foo = tvm.build(s, [A, B], target)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.empty(b_np.shape, dtype=to_dtype, device=dev)
    foo(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np)


if __name__ == "__main__":
    tvm.testing.main()
