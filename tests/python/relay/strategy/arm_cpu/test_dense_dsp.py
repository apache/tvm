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


class BasicDenseTests:
    @tvm.testing.requires_corstone300
    def test_dense(self, shape, weight_shape, dtype, schedule_name):
        """Test a subgraph with a single dense operator."""
        ishape = shape
        wshape = weight_shape
        units = weight_shape[0]
        weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

        input0 = relay.var("input", relay.TensorType(ishape, dtype))
        weight0 = relay.const(weight_data)
        out0 = relay.op.nn.dense(
            input0,
            weight0,
            units=units,
            out_dtype="int32",
        )
        ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

        input1 = relay.var("input", relay.TensorType(ishape, dtype))
        weight1 = relay.const(weight_data)
        out1 = relay.op.nn.dense(
            input1,
            weight1,
            units=units,
            out_dtype="int32",
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


class TestDense(BasicDenseTests):
    """This test is for dense_dsp schedule."""

    shape, weight_shape = tvm.testing.parameters(
        ((1, 128), (16, 128)),
        ((32, 32), (32, 32)),
        ((1, 64), (1, 64)),
        ((11, 2), (2, 2)),
        ((1, 32), (64, 32)),
        ((3, 12), (10, 12)),
    )
    dtype = tvm.testing.parameter("int8", "int16")
    schedule_name = tvm.testing.parameter("dense_dsp")


if __name__ == "__main__":
    tvm.testing.main()
