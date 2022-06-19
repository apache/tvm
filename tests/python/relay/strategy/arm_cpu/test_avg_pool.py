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
import tvm
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import (
    AOT_CORSTONE300_RUNNER,
)


class BasicPoolTests:
    @tvm.testing.requires_corstone300
    def test_pool(
        self,
        pool_type,
        shape,
        dtype,
        pool_size,
        strides,
        padding,
        dilation,
        layout,
        ceil_mode,
        count_include_pad,
        schedule_name,
    ):
        """Test a subgraph with a single pool operator."""
        ishape = shape
        input0 = relay.var("input", relay.TensorType(ishape, dtype))

        out0 = getattr(relay.op.nn, pool_type)(
            input0,
            pool_size=pool_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            layout=layout,
            out_layout="",
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

        ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

        input1 = relay.var("input", relay.TensorType(ishape, dtype))
        out1 = getattr(relay.op.nn, pool_type)(
            input1,
            pool_size=pool_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            layout=layout,
            out_layout="",
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )
        mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

        inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
        output_list = generate_ref_data(ref_mod, inputs)

        compile_and_run(
            AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
            runner=AOT_CORSTONE300_RUNNER,
            interface_api="c",
            use_unpacked_api=True,
            target_opts={
                "-keys": "arm_cpu",
                "-mcpu": "cortex-m7",
            },
            schedule_name=schedule_name,
        )


class TestAvgPool1d(BasicPoolTests):
    """This test is for pool.arm_cpu schedule."""

    (
        shape,
        pool_size,
        strides,
        padding,
        dilation,
        layout,
        ceil_mode,
        count_include_pad,
    ) = tvm.testing.parameters(
        ((3, 32, 27), (3,), (2,), 0, 1, "NCW", False, False),
        ((3, 32, 27), (3,), (2,), 0, 1, "NWC", False, False),
        ((3, 32, 27), (3,), (2,), 0, 1, "NCW", True, False),
        ((3, 32, 27), (3,), (2,), 1, 1, "NCW", False, True),
        ((1, 1, 32), 3, 1, 0, 1, "NCW", False, False),
        ((1, 4, 20), 3, 2, 2, 1, "NCW", False, False),
    )
    pool_type = tvm.testing.parameter("avg_pool1d")
    dtype = tvm.testing.parameter("int32")
    schedule_name = tvm.testing.parameter("pool.arm_cpu")


class TestAvgPool2d(BasicPoolTests):
    """This test is for pool.arm_cpu schedule."""

    (
        shape,
        pool_size,
        strides,
        padding,
        dilation,
        layout,
        ceil_mode,
        count_include_pad,
    ) = tvm.testing.parameters(
        ((3, 32, 27, 27), (3, 3), (2, 2), 0, 1, "NCHW", False, False),
        ((3, 32, 27, 27), (3, 3), (2, 2), 0, 1, "NHWC", False, False),
        ((2, 16, 27, 27), (3, 3), (2, 2), 0, 1, "NCHW", True, False),
        ((2, 27, 27, 16), (3, 3), (2, 2), 0, 1, "NHWC", True, False),
        ((2, 16, 27, 27), (3, 3), (2, 2), 0, 1, "NCHW", True, True),
        ((1, 25, 5, 64), (25, 5), (25, 5), 0, 1, "NHWC", False, False),
        ((1, 3, 3, 256), (3, 3), (3, 3), 0, 1, "NHWC", False, False),
        ((1, 8, 8, 64), (8, 8), (8, 8), 0, 1, "NHWC", False, False),
        ((1, 1, 32, 32), (3, 3), 1, 0, 1, "NCHW", False, False),
        ((1, 4, 32, 20), (3, 3), (2, 2), 0, 1, "NCHW", False, False),
    )
    pool_type = tvm.testing.parameter("avg_pool2d")
    dtype = tvm.testing.parameter("int32")
    schedule_name = tvm.testing.parameter("pool.arm_cpu")


class TestAvgPool3d(BasicPoolTests):
    """This test is for pool.arm_cpu schedule."""

    (
        shape,
        pool_size,
        strides,
        padding,
        dilation,
        layout,
        ceil_mode,
        count_include_pad,
    ) = tvm.testing.parameters(
        ((3, 4, 8, 27, 27), (3, 3, 3), 2, 0, 1, "NCDHW", False, False),
    )
    pool_type = tvm.testing.parameter("avg_pool3d")
    dtype = tvm.testing.parameter("int32")
    schedule_name = tvm.testing.parameter("pool.arm_cpu")


if __name__ == "__main__":
    tvm.testing.main()
