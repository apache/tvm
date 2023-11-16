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
# pylint: disable=invalid-name,missing-docstring
import tvm
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import relax as R


def main():
    class TestModule(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.linear_1 = nn.Linear(in_features, out_features, bias=False)
            self.linear_2 = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: nn.Tensor):
            x1 = self.linear_1(x)
            x2 = self.linear_2(x)
            return x1 + x2

    # pylint: disable=line-too-long
    @I.ir_module
    class ExpectedModule:  # pylint: disable=too-few-public-methods
        @R.function
        def forward(
            x: R.Tensor((1, 10), dtype="float32"),
            packed_params: R.Tuple(
                R.Tensor((20, 10), dtype="float32"), R.Tensor((20, 10), dtype="float32")
            ),
        ) -> R.Tensor((1, 20), dtype="float32"):
            R.func_attr({"num_input": 1})  # type: ignore[attr-defined]
            with R.dataflow():  # type: ignore[attr-defined]
                linear_1_weight: R.Tensor((20, 10), dtype="float32") = packed_params[0]  # type: ignore[valid-type]
                linear_2_weight: R.Tensor((20, 10), dtype="float32") = packed_params[1]  # type: ignore[valid-type]
                permute_dims: R.Tensor((10, 20), dtype="float32") = R.permute_dims(  # type: ignore[attr-defined,valid-type]
                    linear_1_weight, axes=None
                )
                matmul: R.Tensor((1, 20), dtype="float32") = R.matmul(  # type: ignore[attr-defined,valid-type]
                    x, permute_dims, out_dtype="void"
                )
                permute_dims1: R.Tensor((10, 20), dtype="float32") = R.permute_dims(  # type: ignore[attr-defined,valid-type]
                    linear_2_weight, axes=None
                )
                matmul1: R.Tensor((1, 20), dtype="float32") = R.matmul(  # type: ignore[attr-defined,valid-type]
                    x, permute_dims1, out_dtype="void"
                )
                add: R.Tensor((1, 20), dtype="float32") = R.add(matmul, matmul1)  # type: ignore[attr-defined,valid-type]
                gv: R.Tensor((1, 20), dtype="float32") = add  # type: ignore[attr-defined,valid-type]
                R.output(gv)  # type: ignore[attr-defined,valid-type]
            return gv

    # pylint: enable=line-too-long

    model = TestModule(10, 20)
    mod, _ = model.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([1, model.in_features], "float32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            }
        }
    )
    tvm.ir.assert_structural_equal(mod, ExpectedModule)


if __name__ == "__main__":
    main()
