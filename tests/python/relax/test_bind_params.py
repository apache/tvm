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


import tvm
import tvm.script
import tvm.testing
from tvm import relax, tir
from tvm.script import relax as R

import numpy as np
import pytest

param_specification = tvm.testing.parameter("by_string", "by_var")
param_shape = tvm.testing.parameter("static_shape", "dynamic_shape", "ndim", "arbitrary")
tensor_param_dtype = tvm.testing.parameter("float32", None)


def test_bind_tensor_param(param_specification, param_shape, tensor_param_dtype):
    if param_shape == "static_shape":
        shape = [16]
        ndim = -1
    elif param_shape == "dynamic_shape":
        shape = [tir.Var("N", "int64")]
        ndim = -1
    elif param_shape == "ndim":
        shape = None
        ndim = 1
    elif param_shape == "arbitrary":
        shape = None
        ndim = -1
    else:
        raise ValueError(f"Unknown param_shape: {param_shape}")

    @R.function
    def before(A: R.Tensor(shape, ndim=ndim, dtype=tensor_param_dtype)):
        R.func_attr({"global_symbol": "main"})
        B: R.Tensor(shape=shape, ndim=ndim, dtype=tensor_param_dtype) = A
        out = R.add(B, B)
        return out

    np_data = np.arange(16).astype("float32")
    inlined_relax_const = relax.const(np_data)

    @R.function
    def expected() -> R.Tensor([16], "float32"):
        R.func_attr({"global_symbol": "main"})
        B = inlined_relax_const
        out = R.add(B, B)
        return out

    if param_specification == "by_string":
        var = "A"
    elif param_specification == "by_var":
        var = before.params[0]
    else:
        raise ValueError("Unknown param_specification: {param_specification}")

    after = before.bind_params({var: np.arange(16).astype("float32")})

    tvm.ir.assert_structural_equal(expected, after)


def test_bind_shape_param(param_shape):
    if param_shape == "static_shape":
        shape = [16]
        ndim = -1
    elif param_shape == "dynamic_shape":
        shape = [tir.Var("N", "int64")]
        ndim = -1
    elif param_shape == "ndim":
        shape = None
        ndim = 1
    elif param_shape == "arbitrary":
        shape = None
        ndim = -1
    else:
        raise ValueError(f"Unknown param_shape: {param_shape}")

    @R.function
    def before(A: R.Shape(shape, ndim=ndim)):
        R.func_attr({"global_symbol": "main"})
        B: R.Shape(shape, ndim=ndim) = A
        return B

    @R.function
    def expected() -> R.Shape([16]):
        R.func_attr({"global_symbol": "main"})
        B = R.ShapeExpr([16])
        return B

    after = before.bind_params({"A": relax.ShapeExpr([16])})

    tvm.ir.assert_structural_equal(expected, after)


prim_value_dtype = tvm.testing.parameter("int64", "int32", "float32")


def test_bind_prim_value(prim_value_dtype):
    if prim_value_dtype != "int64":
        pytest.xfail(reason="Currently, only support int64 as known symbolic value")

    N = tir.Var("N", prim_value_dtype)
    value = tir.const(16, prim_value_dtype)

    @R.function
    def before(A: R.Prim(value=N)) -> R.Prim(value=N):
        R.func_attr({"global_symbol": "main"})
        B: R.Prim(value=N) = A
        return B

    @R.function
    def expected() -> R.Prim(value=value):
        R.func_attr({"global_symbol": "main"})
        B = R.prim_value(value)
        return B

    after = before.bind_params({"A": relax.PrimValue(value)})

    tvm.ir.assert_structural_equal(expected, after)


def test_error_on_unknown_var():
    @R.function
    def before(A: R.Tensor([16], dtype="float32")):
        R.func_attr({"global_symbol": "main"})
        return A

    unknown_var = relax.Var("unknown_var")

    with pytest.raises(tvm.TVMError):
        before.bind_params({unknown_var: np.arange(16).astype("float32")})


def test_error_on_unknown_var_name():
    @R.function
    def before(A: R.Tensor([16], dtype="float32")):
        R.func_attr({"global_symbol": "main"})
        return A

    with pytest.raises(tvm.TVMError):
        before.bind_params({"unknown_var_name": np.arange(16).astype("float32")})


if __name__ == "__main__":
    tvm.testing.main()
