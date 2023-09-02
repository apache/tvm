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
from tvm.ir import assert_structural_equal
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import relax as R


def test_linear():
    class Activation(nn.Module):
        define_subroutine = True

        def forward(self, state: relax.Expr) -> relax.Var:
            return nn.op.silu(state)

    class Layer(nn.Module):
        define_subroutine = True

        def __init__(self, in_features, out_features):
            self.weights = nn.Parameter((in_features, out_features), dtype="float32")
            self.activation = Activation()

        def forward(self, input: relax.Expr) -> relax.Var:
            state = nn.op.matmul(input, self.weights)
            return self.activation(state)

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            state: R.Tensor(("batch_size", 64), dtype="float32"),
            _io: R.Object,
            weights: R.Tensor((64, 32), dtype="float32"),
        ) -> R.Tuple(R.Tensor(("batch_size", 32), dtype="float32"), R.Tuple(R.Object)):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                state = Expected.layer(state, weights)
                dataflow_output = (state, (_io,))
                R.output(dataflow_output)
            return dataflow_output

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()
                lv: R.Tuple(R.Object) = (_io,)
                gv: R.Tuple(R.Object) = lv
                R.output(gv)

            return gv

        @R.function(private=True)
        def layer(
            state: R.Tensor(("batch_size", 64), dtype="float32"),
            weights: R.Tensor((64, 32), dtype="float32"),
        ) -> R.Tensor(("batch_size", 32), dtype="float32"):
            with R.dataflow():
                state = R.matmul(state, weights)
                state = Expected.activation(state)
                dataflow_output = state
                R.output(dataflow_output)
            return dataflow_output

        @R.function(private=True)
        def activation(
            state: R.Tensor(("batch_size", 32), dtype="float32"),
        ) -> R.Tensor(("batch_size", 32), dtype="float32"):
            with R.dataflow():
                state = R.nn.silu(state)
                dataflow_output = state
                R.output(dataflow_output)
            return dataflow_output

    mod = Layer(64, 32)
    batch_size = tvm.tir.Var("batch_size", "int64")
    tvm_mod, _ = mod.export_tvm(
        spec={"forward": {"input": nn.spec.Tensor((batch_size, 64), "float32")}}, debug=True
    )
    assert_structural_equal(Expected, tvm_mod, True)


if __name__ == "__main__":
    tvm.testing.main()
