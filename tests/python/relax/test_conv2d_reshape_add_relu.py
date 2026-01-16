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
from tvm import relax
from tvm.relax.transform import FuseConv2dReshapeAddRelu
from tvm.script import relax as R


def test_transform_pass():

    # Define the initial IRModule
    @tvm.script.ir_module
    class TestModule:
        @R.function
        def main(
            data: R.Tensor((1, 3, 224, 224), dtype="float32"),
            weight: R.Tensor((64, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((64,), dtype="float32"),
        ):
            with R.dataflow():
                conv_out = R.nn.conv2d(data, weight)
                bias_reshaped = R.reshape(bias, [1, 64, 1, 1])
                bias_add = R.add(conv_out, bias_reshaped)
                relu_out = R.nn.relu(bias_add)
                R.output(relu_out)
            return relu_out

    print(TestModule)

    # Step 1: Apply the FuseConv2dReshapeAddRelu pass
    # This pass identifies the fusion pattern (conv2d-reshape-add-relu)
    # and encapsulates it into a new Relax function with "Composite" attribute.
    fused_mod = FuseConv2dReshapeAddRelu()(TestModule)
    print("=== IR after Step 1 (FuseConv2dReshapeAddRelu) ===")
    print(fused_mod)

    # Step 2: Apply Sequential passes including MergeCompositeFunctions
    # MergeCompositeFunctions takes functions marked with "Composite"
    # and transforms them into functions with a "Codegen" attribute,
    # indicating they should be offloaded to an external backend (e.g., DNNL).
    final_mod = tvm.ir.transform.Sequential(
        [
            relax.transform.FuseConv2dReshapeAddRelu(),
            relax.transform.MergeCompositeFunctions(),
        ]
    )(TestModule)

    print("=== IR after Final Fusion (Sequential Passes) ===")
    print(final_mod)

    # Check attributes of functions in the final module
    # This helps confirm if "Codegen" attribute was successfully added to the fused function.
    print("=== Function Attributes in Final IR ===")
    for name, func in final_mod.functions.items():
        if hasattr(func, "attrs") and func.attrs:
            print(f"Function {name} attributes:", func.attrs)


if __name__ == "__main__":
    test_transform_pass()
