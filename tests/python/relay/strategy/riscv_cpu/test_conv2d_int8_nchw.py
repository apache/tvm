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
import re
import tvm
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import AOTTestRunner


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
        schedule_name,
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

        target_opts = {
            "-keys": "riscv_cpu",
            "-march": "rv64gcv",
        }

        def checker(base_path: str) -> bool:
            def read_file(path):
                with open(path) as f:
                    return f.read()

            default_lib1 = read_file(base_path + "/codegen/host/src/default_lib1.c")
            regex = r"(?s)dot_uint8_int8_int32_update(.*?)"
            return re.search(regex, default_lib1) is not None

        assert compile_and_run(
            AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
            runner=AOTTestRunner(makefile="riscv"),
            interface_api="c",
            use_unpacked_api=True,
            target_opts=target_opts,
            schedule_name=schedule_name,
            checker=checker,
        )


class TestConv2d_NCHW(RISCVConv2dInt8):
    (data_shape, kernel_size, num_filter,) = tvm.testing.parameters(
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
    schedule_name = tvm.testing.parameter("conv2d_int8_NCHW.riscv_cpu")


if __name__ == "__main__":
    tvm.testing.main()
