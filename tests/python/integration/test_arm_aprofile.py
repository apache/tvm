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
"""Tests for Arm(R) A-Profile Architecture."""
import os
import subprocess

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.transform import ToMixedPrecision, FoldConstant
from tvm.relay.build_module import bind_params_by_name
from tvm.testing.aot import AOTTestModel, AOTTestRunner, generate_ref_data, compile_and_run
from tvm.contrib import utils


def get_mattr(dtype):
    mattr = "+v8.2a,+neon"
    if dtype == "float16":
        mattr += ",+fullfp16"
    elif dtype == "bfloat16":
        mattr += ",+bf16"
    return mattr


@tvm.testing.skip_if_32bit(reason="skipping test for i386.")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_conv2d(dtype):
    """Test if Conv2d cross compiles with TVM schedules."""
    dtype = "float32"
    ishape = [1, 28, 28, 3]  # NHWC
    kernel_size = (3, 3)
    wshape = (kernel_size[0], kernel_size[1], ishape[-1], 2)  # HWIO
    weight_data = np.random.uniform(-128, 127, wshape).astype(dtype)
    invar = relay.var("data", relay.TensorType(ishape, dtype))
    weight = relay.const(weight_data, dtype)
    out = relay.op.nn.conv2d(
        invar,
        weight,
        kernel_size=kernel_size,
        channels=2,
        strides=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype=dtype,
        out_layout="NHWC",
    )
    mod = tvm.IRModule.from_expr(relay.Function([invar], out))
    params = {}

    prefixed_network_name = dtype + ".conv2d"
    lib_path = os.getcwd() + "/" + prefixed_network_name + ".mod.so"
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=" + get_mattr(dtype)

    mod["main"] = bind_params_by_name(mod["main"], params)
    if dtype in ["float16", "bfloat16"]:
        mod = ToMixedPrecision(dtype)(mod)
        mod = FoldConstant()(mod)

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target=target, params=params)
        lib.export_library(lib_path, cc="aarch64-linux-gnu-gcc")


# AOT Test Runner using the AArch64 Architecture Envelope Model (AEM)
# Fixed Virtual Platform (FVP) reference system.
# See: https://developer.arm.com/Tools%20and%20Software/Fixed%20Virtual%20Platforms
AOT_APROFILE_AEM_RUNNER = AOTTestRunner(
    makefile="aprofile_aem",
    pass_config={
        "tir.usmp.enable": False,
        "tir.disable_assert": True,  # AOT test infra creates 'fake' inputs that fail asserts
    },
)


@tvm.testing.requires_x86
def test_aem_simple_addition():
    """Tests a simple addition running on the AArch64 AEM."""
    inp = relay.var("data", shape=(1, 2, 4, 4))
    add = relay.add(inp, relay.const(np.ones((1, 2, 4, 4))))
    func = relay.Function([inp], add)
    ir_mod = tvm.IRModule.from_expr(func)
    ir_mod = tvm.relay.transform.InferType()(ir_mod)

    main_func = ir_mod["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    input_data = np.random.uniform(size=shape_dict["data"]).astype(type_dict["data"])
    params = {}
    inputs = {"data": input_data}
    ref_outputs = generate_ref_data(ir_mod, inputs, params)

    compile_and_run(
        AOTTestModel(module=ir_mod, inputs=inputs, outputs=ref_outputs, params=params),
        target=tvm.target.Target("llvm -mtriple=aarch64-none-elf"),
        runtime=tvm.relay.backend.Runtime("crt", {"system-lib": True}),
        interface_api="packed",
        use_unpacked_api=False,
        runner=AOT_APROFILE_AEM_RUNNER,
    )


@tvm.testing.requires_x86
def test_aem_asm_sme():
    """
    Tests SME assembly runs on the AArch64 AEM. This test is used as a simple
    sanity check until the TVM schedules are able to produce SME.
    """
    c_code = """
    #include <stdio.h>

    int main(void) {
        __asm volatile(
            "smstart\\n"
            "smstop\\n"
        );
        printf("EXITTHESIM\\n");
        return 0;
    }
    """
    runner = AOT_APROFILE_AEM_RUNNER

    tmpdir = utils.tempdir()
    build_path = os.path.join(tmpdir.path, "build")
    os.makedirs(build_path, exist_ok=True)

    with open(build_path + "/test.c", "w") as f:
        f.write(c_code)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    makefile_dir = os.path.join(file_dir, "../../../tests/python/relay/aot")
    makefile = os.path.join(makefile_dir, f"{runner.makefile}.mk")

    make_command = (
        f"make -f {makefile} build_dir={build_path}"
        + f" TVM_ROOT={file_dir}/../../.."
        + f" AOT_TEST_ROOT={makefile_dir}"
        + " FVP_DIR=/opt/arm/fvp/Base_RevC_AEMvA_pkg/models/Linux64_GCC-9.3/"
    )

    compile_command = f"{make_command} aot_test_runner"
    popen = subprocess.Popen(compile_command, cwd=build_path, shell=True, stdout=subprocess.PIPE)
    return_code = popen.wait()
    assert not return_code, "Failed to compile"

    run_command = f"{make_command} run"
    popen = subprocess.Popen(run_command, cwd=build_path, shell=True, stdout=subprocess.PIPE)
    return_code = popen.wait()
    assert not return_code, "Failed to run"


if __name__ == "__main__":
    tvm.testing.main()
