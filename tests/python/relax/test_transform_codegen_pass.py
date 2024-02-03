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

import pytest
import os
import tvm
import tvm.testing
from tvm import relax, tir
import numpy as np
from tvm.script import relax as R, ir as I, tir as T
from tvm.relax.testing import transform
import tempfile
from tvm.relax.transform.tuning_api import Trace
from tvm.relax.dpl import is_op, wildcard

env_checker_codegen = tvm.get_global_func("relax.ext.tensorrt", True)
env_checker_runtime = tvm.get_global_func("relax.is_tensorrt_runtime_enabled", True)

has_tensorrt_codegen = pytest.mark.skipif(
    not env_checker_codegen,
    reason="TensorRT codegen not available",
)
has_tensorrt_runtime = pytest.mark.skipif(
    not env_checker_runtime or not env_checker_runtime(),
    reason="TensorRT runtime not available",
)

# Global variable in pytest that applies markers to all tests.
pytestmark = [has_tensorrt_codegen, has_tensorrt_runtime]

# Target gpu
target_str = "nvidia/nvidia-t4"
target = tvm.target.Target(target_str)
dev = tvm.cuda()


def check_executable(exec, dev, inputs, expected, entry_func_name):
    vm = relax.VirtualMachine(exec, dev)
    out = vm[entry_func_name](*inputs)
    tvm.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)


def check_roundtrip(exec0, dev, inputs, expected, entry_func_name="main"):
    exec0.mod.export_library("exec.so")
    exec1 = tvm.runtime.load_module("exec.so")
    os.remove("exec.so")
    assert exec0.stats() == exec1["stats"]()
    assert exec0.as_text() == exec1["as_text"]()

    check_executable(exec0, dev, inputs, expected, entry_func_name)
    check_executable(exec1, dev, inputs, expected, entry_func_name)


def gen_ground_truth(mod, target, dev, inputs):
    # Lower and run tuning
    # Since there is no default schedule for GPU in MS yet, this is necessary
    with target:
        seq = tvm.transform.Sequential(
            [relax.transform.LegalizeOps(), tir.transform.DefaultGPUSchedule()]
        )
        new_mod = seq(mod)
    assert relax.analysis.well_formed(new_mod)
    exec = relax.build(new_mod, target, params={})
    vm = relax.VirtualMachine(exec, dev)
    return vm["main"](*inputs)


@tvm.script.ir_module
class InputModule:
    @R.function
    def main(
        x: R.Tensor((16, 16), "float32"), y: R.Tensor((16, 16), "float32")
    ) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            z1 = R.multiply(x, y)
            z2 = R.add(z1, x)
            z3 = R.add(z1, z2)
            z4 = R.multiply(z3, z2)
            z5 = R.add(z4, z1)
            R.output(z5)
        return z5


def setup_test():
    # Prepare IRModule and its input
    mod = InputModule
    assert isinstance(mod, tvm.IRModule)

    np0 = np.random.rand(16, 16).astype(np.float32)
    np1 = np.random.rand(16, 16).astype(np.float32)
    data0 = tvm.nd.array(np0, dev)
    data1 = tvm.nd.array(np1, dev)
    inputs = [data0, data1]

    # Ground truth should be generated before annotation
    # due to the conflict with MS task extraction
    # TODO(@sunggg): Sort this out
    expected = gen_ground_truth(mod, target, dev, inputs)
    return mod, inputs, expected


entry_func_name = tvm.testing.parameter("main", "func")


@tvm.testing.requires_gpu
def test_tensorrt_only(entry_func_name):
    mod, inputs, expected = setup_test()

    if entry_func_name != "main":
        mod[entry_func_name] = mod
        del mod["main"]

    # Define patterns that we want to offload to byoc
    # This test will offload entire model
    # Thus, define patterns for both `multiply` and `add` ops
    patterns = [
        ("tensorrt.multiply", is_op("relax.multiply")(wildcard(), wildcard())),
        ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
    ]

    new_mod = tvm.transform.Sequential(
        [
            relax.transform.FuseOpsByPattern(patterns),
            relax.transform.MergeCompositeFunctions(),
            relax.transform.RunCodegen(),
        ]
    )(mod)

    ex0 = relax.build(new_mod, target, params={})
    # Sanity check for the correctness and roundtrip
    check_roundtrip(ex0, dev, inputs, expected, entry_func_name)


@tvm.testing.requires_gpu
def test_mix_use_tensorrt_and_tvm():
    mod, inputs, expected = setup_test()

    # Define patterns that we want to offload to byoc
    # This test will only offload `add` op to tensorrt
    # and tune `multiply` op with MetaSchedule
    patterns = [
        ("tensorrt.add", is_op("relax.add")(wildcard(), wildcard())),
    ]

    # Run Codegen pass
    with tempfile.TemporaryDirectory() as work_dir:
        with target, tvm.transform.PassContext(trace=Trace(mod), opt_level=0):
            new_mod = tvm.transform.Sequential(
                [
                    relax.transform.FuseOpsByPattern(patterns),
                    relax.transform.MergeCompositeFunctions(),
                    relax.transform.RunCodegen(),
                    relax.transform.LegalizeOps(),
                    relax.transform.MetaScheduleTuneIRMod(
                        params={}, work_dir=work_dir, max_trials_global=8
                    ),
                    relax.transform.MetaScheduleApplyDatabase(work_dir),
                ]
            )(mod)
    assert relax.analysis.well_formed(new_mod)
    with transform.PassContext(opt_level=0):
        ex0 = relax.build(new_mod, target, params={})

    # Sanity check for the correctness and roundtrip
    check_roundtrip(ex0, dev, inputs, expected)


@tvm.script.ir_module
class Conv2dx2:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        weight2: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        cls = Conv2dx2
        with R.dataflow():
            lv: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_tensorrt(
                data, weight1
            )
            gv: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_tensorrt(
                lv, weight2
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_tensorrt(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        R.func_attr({"Codegen": "tensorrt", "global_symbol": "fused_relax_nn_conv2d_tensorrt"})

        @R.function
        def gv(
            data_1: R.Tensor((16, 32, 32, 16), dtype="float16"),
            weight1_1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
            R.func_attr({"Composite": "tensorrt.conv2d", "Primitive": 1})
            with R.dataflow():
                gv_1: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(
                    data_1,
                    weight1_1,
                    padding=[1, 1, 1, 1],
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                )
                R.output(gv_1)
            return gv_1

        gv1: R.Tensor((16, 32, 32, 16), dtype="float16") = gv(data, weight1)
        return gv1


@tvm.script.ir_module
class Conv2dx2_after:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        weight2: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        with R.dataflow():
            lv = R.call_dps_packed(
                "fused_relax_nn_conv2d_tensorrt",
                (data, weight1),
                out_sinfo=R.Tensor((16, 32, 32, 16), dtype="float16"),
            )
            gv = R.call_dps_packed(
                "fused_relax_nn_conv2d_tensorrt",
                (lv, weight2),
                out_sinfo=R.Tensor((16, 32, 32, 16), dtype="float16"),
            )
            R.output(gv)
        return gv


def test_multiple_calls_same_extern():
    mod = relax.transform.RunCodegen()(Conv2dx2)
    tvm.ir.assert_structural_equal(mod["main"], Conv2dx2_after["main"])


def test_default_entry_func():
    """The entry function is not necessarily named "main"

    Like `test_multiple_calls_same_extern`, but the main function is
    named "func".
    """
    before_with_main = Conv2dx2
    after_with_main = relax.transform.RunCodegen()(before_with_main)

    def rename_main(mod):
        mod = mod.clone()
        mod["func"] = mod["main"].with_attr("global_symbol", "func")
        del mod["main"]
        return mod

    before_with_func = rename_main(before_with_main)
    expected_with_func = rename_main(after_with_main)
    after_with_func = relax.transform.RunCodegen()(before_with_func)

    tvm.ir.assert_structural_equal(expected_with_func["func"], after_with_func["func"])


def test_dynamic_shape():
    import tvm.relax.backend.contrib.cublas

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 4096), dtype="float16"),
            w1: R.Tensor((4096, "r1"), dtype="float16"),
            w2: R.Tensor((4096, "r2"), dtype="float16"),
        ) -> R.Tuple(R.Tensor((1, "r1"), dtype="float16"), R.Tensor((1, "r2"), dtype="float16")):
            r1 = T.int64()
            r2 = T.int64()
            cls = Before
            with R.dataflow():
                lv: R.Tensor((1, r1), dtype="float16") = cls.fused_relax_matmul_cublas(x, w1)
                lv1: R.Tensor((1, r2), dtype="float16") = cls.fused_relax_matmul_cublas(x, w2)
                gv: R.Tuple(
                    R.Tensor((1, r1), dtype="float16"), R.Tensor((1, r2), dtype="float16")
                ) = (lv, lv1)
                R.output(gv)
            return gv

        @R.function
        def fused_relax_matmul_cublas(
            x: R.Tensor((1, 4096), dtype="float16"), w1: R.Tensor((4096, "r1"), dtype="float16")
        ) -> R.Tensor((1, "r1"), dtype="float16"):
            r1 = T.int64()
            R.func_attr({"Codegen": "cublas"})

            @R.function
            def gv(
                x_1: R.Tensor((1, 4096), dtype="float16"),
                w1_1: R.Tensor((4096, r1), dtype="float16"),
            ) -> R.Tensor((1, r1), dtype="float16"):
                R.func_attr({"Composite": "cublas.matmul"})
                with R.dataflow():
                    gv_1: R.Tensor((1, r1), dtype="float16") = R.matmul(x_1, w1_1, out_dtype="void")
                    R.output(gv_1)
                return gv_1

            gv1: R.Tensor((1, r1), dtype="float16") = gv(x, w1)
            return gv1

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 4096), dtype="float16"),
            w1: R.Tensor((4096, "r1"), dtype="float16"),
            w2: R.Tensor((4096, "r2"), dtype="float16"),
        ) -> R.Tuple(R.Tensor((1, "r1"), dtype="float16"), R.Tensor((1, "r2"), dtype="float16")):
            r1 = T.int64()
            r2 = T.int64()
            with R.dataflow():
                lv = R.call_dps_packed(
                    "fused_relax_matmul_cublas",
                    (x, w1),
                    out_sinfo=R.Tensor((1, r1), dtype="float16"),
                )
                lv1 = R.call_dps_packed(
                    "fused_relax_matmul_cublas",
                    (x, w2),
                    out_sinfo=R.Tensor((1, r2), dtype="float16"),
                )
                gv: R.Tuple(
                    R.Tensor((1, r1), dtype="float16"), R.Tensor((1, r2), dtype="float16")
                ) = (lv, lv1)
                R.output(gv)
            return gv

    after = relax.transform.RunCodegen()(Before)
    tvm.ir.assert_structural_equal(after["main"], Expected["main"])


# TODO(@sunggg):  test with more complex patterns (e.g., multiple annots, mixed codegens, different ops, const binding)

if __name__ == "__main__":
    pytest.main([__file__])
