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

# TODO remove
import sys

import torch
from torch import fx
from torch.nn import Module

sys.path.append("/ssd1/htalendr/tvm/python")  # Refer to local TVM build


import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.torch import from_fx
from tvm.script import relax as R


def verify_model(torch_model, input_info, binding, expected):
    graph_model = fx.symbolic_trace(torch_model)
    with torch.no_grad():
        mod = from_fx(graph_model, input_info)
    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm.ir.assert_structural_equal(mod, expected)


def test_masked_scatter():
    class MaskedScatter1(Module):
        def forward(self, data, mask, src):
            return data.masked_scatter(mask, src)

    class MaskedScatter2(Module):
        def forward(self, data, mask, src):
            return data.masked_scatter(mask, src)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"),
            inp_1: R.Tensor((5,), dtype="bool"),
            inp_2: R.Tensor((10,), dtype="float32"),
        ) -> R.Tensor((5,), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="int32") = R.cumsum(
                    inp_1, axis=0, dtype="int32", exclusive=False
                )
                lv1: R.Tensor((5,), dtype="int32") = R.subtract(lv, R.const(1, "int32"))
                lv2: R.Tensor((5,), dtype="float32") = R.take(inp_2, lv1, axis=0)
                lv3: R.Tensor((5,), dtype="float32") = R.where(inp_1, lv2, inp_0)
                gv: R.Tensor((5,), dtype="float32") = lv3
                R.output(gv)
            return gv

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 5), dtype="float32"),
            inp_1: R.Tensor((2, 5), dtype="bool"),
            inp_2: R.Tensor((3, 5), dtype="float32"),
        ) -> R.Tensor((2, 5), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10,), dtype="bool") = R.reshape(inp_1, R.shape([10]))
                lv1: R.Tensor((10,), dtype="int32") = R.cumsum(
                    lv, axis=0, dtype="int32", exclusive=False
                )
                lv2: R.Tensor((10,), dtype="int32") = R.subtract(lv1, R.const(1, "int32"))
                lv3: R.Tensor((15,), dtype="float32") = R.reshape(inp_2, R.shape([15]))
                lv4: R.Tensor((10,), dtype="float32") = R.take(lv3, lv2, axis=0)
                lv5: R.Tensor((2, 5), dtype="float32") = R.reshape(lv4, R.shape([2, 5]))
                lv6: R.Tensor((2, 5), dtype="float32") = R.where(inp_1, lv5, inp_0)
                gv: R.Tensor((2, 5), dtype="float32") = lv6
                R.output(gv)
            return gv

    verify_model(
        MaskedScatter1(), [([5], "float32"), ([5], "bool"), ([10], "float32")], {}, expected1
    )
    verify_model(
        MaskedScatter2(),
        [([2, 5], "float32"), ([2, 5], "bool"), ([3, 5], "float32")],
        {},
        expected2,
    )


test_masked_scatter()

# if __name__ == "__main__":
#     tvm.testing.main()
