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
import tvm.testing
from tvm import relax
from tvm.relax.testing import nn
from tvm.script import ir as I, relax as R


def test_emit():
    class ReLU(nn.Module):
        def forward(self, input: relax.Expr) -> relax.Var:
            return nn.emit(relax.op.nn.relu(input))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((32, 32), dtype="float32")) -> R.Tensor((32, 32), dtype="float32"):
            gv: R.Tensor((32, 32), dtype="float32") = R.nn.relu(x)
            return gv

    bb = relax.BlockBuilder()
    with bb.function("main"):
        model = ReLU()
        x = nn.Placeholder((32, 32), dtype="float32", name="x")
        output = model(x)
        params = [x] + model.parameters()
        bb.emit_func_output(output, params)

    tvm.ir.assert_structural_equal(bb.get(), Expected)


def test_get_param():
    class Plus1(nn.Module):
        def __init__(self):
            self.const_1 = relax.const(1, "float32")

        def forward(self, input: relax.Expr) -> relax.Var:
            return nn.emit(relax.op.add(input, self.const_1))

    model = Plus1()
    assert model.parameters() == []


if __name__ == "__main__":
    tvm.testing.main()
