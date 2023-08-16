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

import pathlib

import numpy as np

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.testing.aot import generate_ref_data


def _make_session(temp_dir, mod):
    template_project_dir = pathlib.Path(tvm.micro.get_microtvm_template_projects("riscv"))
    options = {
        "toolchain_path": "/opt/riscv",
        "target": "riscv64-unknown-linux-gnu",
        "march": "rv64gcv",
        "verbose": 1,
    }
    project = tvm.micro.generate_project(template_project_dir, mod, temp_dir / "project", options)
    project.build()
    project.flash()
    return tvm.micro.Session(project.transport())


class RISCVConv2dInt8:
    @tvm.testing.requires_riscv_spike
    def test_conv2d_int8(
        self,
        data_shape,
        kernel_size,
        data_layout,
        kernel_layout,
        num_filter,
        padding,
        dtype,
        wtype,
    ):
        weight_shape = (num_filter, data_shape[1], *kernel_size)

        data = relay.var("input", shape=data_shape, dtype=dtype)

        if "int" in wtype:
            min_w_value = np.iinfo(wtype).min
            max_w_value = np.iinfo(wtype).max
        else:
            min_w_value = np.finfo(wtype).min
            max_w_value = np.finfo(wtype).max

        weight_data = np.random.randint(
            low=min_w_value, high=max_w_value, size=weight_shape, dtype=wtype
        )
        weight = relay.const(weight_data)

        func = relay.qnn.op.conv2d(
            data,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=weight_shape[0],
            kernel_size=kernel_size,
            padding=padding,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )

        ref_mod = relay.Function(relay.analysis.free_vars(func), func)
        ref_mod = tvm.IRModule.from_expr(ref_mod)

        if "int" in dtype:
            min_d_value = np.iinfo(dtype).min
            max_d_value = np.iinfo(dtype).max
        else:
            min_d_value = np.finfo(dtype).min
            max_d_value = np.finfo(dtype).max

        inputs = {
            "input": np.random.randint(
                low=min_d_value, high=max_d_value, size=data_shape, dtype=dtype
            )
        }

        output_list = generate_ref_data(ref_mod, inputs)

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(mod)

        temp_dir = tvm.contrib.utils.tempdir()
        target = "c -keys=riscv_cpu -march=rv64gcv"
        target = tvm.target.Target(target, host="c")
        runtime = Runtime("crt", {"system-lib": True})
        executor = Executor("aot")
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            factory = tvm.relay.build(mod, target=target, runtime=runtime, executor=executor)

        def do_test():
            aot_executor = tvm.micro.create_local_aot_executor(sess)
            aot_executor.get_input("input").copyfrom(inputs["input"])
            aot_executor.run()

            out = aot_executor.get_output(0)
            assert (out.numpy() == output_list["output"]).all()

        with _make_session(temp_dir, factory) as sess:
            do_test()


class TestConv2d_NCHW(RISCVConv2dInt8):
    (
        data_shape,
        kernel_size,
        num_filter,
    ) = tvm.testing.parameters(
        ((1, 128, 14, 14), (3, 3), 128),
        ((1, 128, 14, 14), (1, 1), 256),
        ((1, 256, 7, 7), (1, 1), 512),
        ((1, 256, 7, 7), (3, 3), 512),
        ((1, 512, 3, 3), (3, 3), 512),
    )
    padding = tvm.testing.parameter((1, 1))
    data_layout = tvm.testing.parameter("NCHW")
    kernel_layout = tvm.testing.parameter("OIHW")
    dtype = tvm.testing.parameter("uint8")
    wtype = tvm.testing.parameter("int8")


if __name__ == "__main__":
    tvm.testing.main()
