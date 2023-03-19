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

from typing import Union

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.relax import Function
from tvm.script import relax as R, tir as T


def _check(before: Union[Function, IRModule], expected: Union[Function, IRModule]):
    if isinstance(before, Function):
        before = IRModule({"main": before})
    if isinstance(expected, Function):
        expected = IRModule({"main": expected})
    after = relax.transform.DecomposeCompositeOps()(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_batch_norm_simple():
    @R.function
    def before(
        x: R.Tensor((1, 64, 112, 112), "float32"),
        gamma: R.Tensor((64,), "float32"),
        beta: R.Tensor((64,), "float32"),
        moving_mean: R.Tensor((64,), "float32"),
        moving_var: R.Tensor((64,), "float32"),
    ):
        with R.dataflow():
            bn = R.nn.batch_norm(
                x,
                gamma,
                beta,
                moving_mean,
                moving_var,
                axis=1,
                epsilon=1e-5,
                center=True,
                scale=True,
            )
            gv = bn[0]
            R.output(gv)
        return gv

    @R.function
    def expected(
        x: R.Tensor((1, 64, 112, 112), "float32"),
        gamma: R.Tensor((64,), "float32"),
        beta: R.Tensor((64,), "float32"),
        moving_mean: R.Tensor((64,), "float32"),
        moving_var: R.Tensor((64,), "float32"),
    ):
        with R.dataflow():
            mean = R.expand_dims(moving_mean, axis=[0, 2, 3])
            out = x - mean
            var = R.expand_dims(moving_var, axis=[0, 2, 3])
            var_eps = var + R.const(1e-05, "float32")
            sqrt_var = R.sqrt(var_eps)
            div = R.divide(out, sqrt_var)
            new_gamma = R.expand_dims(gamma, axis=[0, 2, 3])
            out = div * new_gamma
            new_beta = R.expand_dims(beta, axis=[0, 2, 3])
            out = out + new_beta
            R.output(out)
        return out

    _check(before, expected)


def test_batch_norm_complex():
    @R.function
    def before(
        x: R.Tensor((1, 64, 112, 112), "float32"),
        gamma: R.Tensor((64,), "float32"),
        beta: R.Tensor((64,), "float32"),
        moving_mean: R.Tensor((64,), "float32"),
        moving_var: R.Tensor((64,), "float32"),
    ):
        with R.dataflow():
            bn = R.nn.batch_norm(
                x,
                gamma,
                beta,
                moving_mean,
                moving_var,
                axis=1,
                epsilon=1e-5,
                center=True,
                scale=True,
            )
            gv0 = bn[0]
            gv1 = bn[1]
            R.output(gv0, gv1)
        return gv0, gv1

    @R.function
    def expected(
        x: R.Tensor((1, 64, 112, 112), "float32"),
        gamma: R.Tensor((64,), "float32"),
        beta: R.Tensor((64,), "float32"),
        moving_mean: R.Tensor((64,), "float32"),
        moving_var: R.Tensor((64,), "float32"),
    ):
        with R.dataflow():
            # bn[1] is used, so we need to keep the original batch_norm
            # NOTE: It's a rare case, so that we don't optimize it for now
            bn = R.nn.batch_norm(
                x,
                gamma,
                beta,
                moving_mean,
                moving_var,
                axis=1,
                epsilon=1e-5,
                center=True,
                scale=True,
            )
            mean = R.expand_dims(moving_mean, axis=[0, 2, 3])
            out = x - mean
            var = R.expand_dims(moving_var, axis=[0, 2, 3])
            var_eps = var + R.const(1e-05, "float32")
            sqrt_var = R.sqrt(var_eps)
            div = R.divide(out, sqrt_var)
            new_gamma = R.expand_dims(gamma, axis=[0, 2, 3])
            out = div * new_gamma
            new_beta = R.expand_dims(beta, axis=[0, 2, 3])
            out = out + new_beta
            gv1 = bn[1]
            R.output(out, gv1)
        return out, gv1

    _check(before, expected)


def test_op_tensor_to_shape():
    @R.function
    def before(t: R.Tensor(ndim=1, dtype="int64")):
        gv: R.Shape(ndim=3) = R.tensor_to_shape(t)
        return gv

    @R.function
    def expected(t: R.Tensor(dtype="int64", ndim=1)) -> R.Shape(ndim=3):
        x = T.int64()
        x_1 = T.int64()
        x_2 = T.int64()
        gv: R.Shape(ndim=3) = R.call_packed(
            "vm.builtin.tensor_to_shape", t, sinfo_args=(R.Shape(ndim=3),)
        )
        y: R.Shape([x, x_1, x_2]) = R.match_cast(gv, R.Shape([x, x_1, x_2]))
        gv_1: R.Shape([x, x_1, x_2]) = R.shape([x, x_1, x_2])
        return gv_1

    _check(before, expected)


if __name__ == "__main__":
    tvm.testing.main()
