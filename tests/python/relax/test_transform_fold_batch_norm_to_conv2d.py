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

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.ir.module import IRModule
from tvm.script.ir_builder import relax as relax_builder
from tvm.relax.expr_functor import PyExprVisitor, visitor


def get_conv2d_batchnorm_sample():
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            data = R.arg("data", R.Tensor((1, 3, 224, 224), "float32"))
            weight = R.arg("weight", R.Tensor((32, 3, 3, 3), "float32"))
            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d(
                        data,
                        weight,
                        out_dtype="float32",
                        strides=(1, 1),
                        dilation=(1, 1),
                        padding=(1, 1),
                        data_layout="NCHW",
                        kernel_layout="OIHW",
                        groups=1,
                    )
                )
                gamma = R.arg("gamma", R.Tensor((32,), "float32"))
                beta = R.arg("beta", R.Tensor((32,), "float32"))
                mean = R.arg("mean", R.Tensor((32,), "float32"))
                variance = R.arg("variance", R.Tensor((32,), "float32"))
                output = R.emit(
                    R.nn.batch_norm(output, gamma, beta, mean, variance, axis=1, epsilon=1e-5)[0]
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()

    return tvm.IRModule({"main": func})


def test_fold_batchnorm_info_conv2d():
    mod = get_conv2d_batchnorm_sample()
    mod_fold = get_conv2d_batchnorm_sample()

    target = tvm.target.Target("llvm", host="llvm")
    data_in = tvm.nd.array(np.random.rand(1, 3, 224, 224).astype(np.float32))

    weight_data = tvm.nd.array(np.random.rand(32, 3, 3, 3).astype(np.float32))
    gamma_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    beta_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    mean_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    variance_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    params_np = {
        "weight": weight_data,
        "gamma": gamma_data,
        "beta": beta_data,
        "mean": mean_data,
        "variance": variance_data,
    }

    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    mod_fold = tvm.relax.transform.BindParams("main", params_np)(mod_fold)

    # Normal build
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = vm["main"](data_in)

    # Fold BN to Conv2D
    mod_fold = relax.transform.FoldBatchnormToConv2D()(mod_fold)
    mod_fold = relax.transform.FoldConstant()(mod_fold)
    ex_fold = relax.build(mod_fold, target)
    vm_fold = relax.VirtualMachine(ex_fold, tvm.cpu())
    out_fold = vm_fold["main"](data_in)

    tvm.testing.assert_allclose(out.numpy(), out_fold.numpy(), rtol=1e-5, atol=1e-5)


@visitor
class VerifyFolding(PyExprVisitor):  # pylint: disable=abstract-method
    def visit(self, mod: IRModule) -> None:
        """Entry point"""
        for _, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.visit_expr(func)

    def visit_call_(self, call: relax.Call) -> None:  # pylint: disable=arguments-renamed
        assert (
            call.op.name != "relax.nn.batch_norm"
        ), f"Batchnorm op shouldn't be present after folding to previous conv2d"


def test_fold_batchnorm_info_conv2d_transform():
    mod = get_conv2d_batchnorm_sample()
    mod = relax.transform.FoldBatchnormToConv2D()(mod)
    weight_data = tvm.nd.array(np.random.rand(32, 3, 3, 3).astype(np.float32))
    gamma_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    beta_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    mean_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    variance_data = tvm.nd.array(np.random.rand(32).astype(np.float32))
    params_np = {
        "weight": weight_data,
        "gamma": gamma_data,
        "beta": beta_data,
        "mean": mean_data,
        "variance": variance_data,
    }
    mod = tvm.relax.transform.BindParams("main", params_np)(mod)
    mod = relax.transform.FoldBatchnormToConv2D()(mod)
    mod = relax.transform.FoldConstant()(mod)

    VerifyFolding().visit(mod)


if __name__ == "__main__":
    tvm.testing.main()
