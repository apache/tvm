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

from tvm import relax, tir
from tvm.ir import assert_structural_equal
from tvm.relax.frontend import nn
from tvm.script import ir as I, relax as R, tir as T


def test_simple():
    """A module may be exported from nn.Module to Relax"""

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((3, 3), "float32")}},
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor([3, 3], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_custom_module():
    """A module may be exported from nn.Module to Relax"""

    class Before(nn.Module):
        def forward(self, x: R.Tensor):
            return nn.op.relu(x)

    slm_mod = Before()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((3, 3), "float32")}},
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor([3, 3], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_debug_effect():
    """Passing debug=True provides an argument for IO effect"""

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((3, 3), "float32")}},
        debug=True,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            x: R.Tensor([3, 3], dtype="float32"),
            _io: R.Object,
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                relu = R.nn.relu(x)
                output = relu, (_io,)
                R.output(output)
            return output

        @R.function
        def _initialize_effect():
            with R.dataflow():
                _io = R.null_value()
                output = (_io,)
                R.output(output)
            return output

    assert_structural_equal(exported_mod, Expected)


def test_struct_info_specification():
    """An argument may be specified with relax StructInfo"""

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": relax.TensorStructInfo([3, 3], "float32")}},
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor([3, 3], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_tvmscript_struct_info_specification():
    """An argument may be specified with R.Tensor

    The same syntax used in TVMScript for type annotations may be used
    when exporting from SLM.
    """

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": R.Tensor([3, 3], "float32")}},
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor([3, 3], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_dynamic_shape():
    """An argument may have a dynamic shape"""

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor([tir.Var("batch_size", "int64"), 8], "float32")}},
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor(["batch_size", 8], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_dynamic_shape_in_multiple_functions():
    """A dynamic shape may be used in multiple functions"""

    class Before(nn.Module):
        def forward_relu(self, x: nn.Tensor):
            return nn.relu(x)

        def forward_silu(self, x: nn.Tensor):
            return nn.silu(x)

    slm_mod = Before()
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward_relu": {"x": nn.spec.Tensor((tir.Var("batch_size", "int64"), 8), "float32")},
            "forward_silu": {"x": nn.spec.Tensor((tir.Var("batch_size", "int64"), 8), "float32")},
        },
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward_relu(x: R.Tensor(["batch_size", 8], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                R.output(relu)
            return relu

        @R.function
        def forward_silu(x: R.Tensor(["batch_size", 8], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                silu = R.nn.silu(x)
                R.output(silu)
            return silu

    assert_structural_equal(exported_mod, Expected)


def test_export_nested_module():
    """nn.Module instances may contain other nn.Module

    When exporting to a Relax IRModule, all `nn.Parameter` instances
    within the `nn.Module` become Relax function parameters.
    """

    class LlamaMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.gate_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.down_proj = nn.Linear(
                intermediate_size,
                hidden_size,
                dtype="float16",
                bias=False,
            )

        def forward(self, x: nn.Tensor):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(nn.op.silu(gate) * up)

    hidden_size = 4096
    intermediate_size = 11008
    slm_mod = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor((tir.Var("batch_size", "int64"), hidden_size), "float16")
            },
        },
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            x: R.Tensor(["batch_size", hidden_size], "float16"),
            gate_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            up_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            down_proj_weights: R.Tensor([hidden_size, intermediate_size], "float16"),
        ):
            R.func_attr({"num_input": 1})
            batch_size = T.int64()
            with R.dataflow():
                gate: R.Tensor([batch_size, intermediate_size]) = R.matmul(
                    x, R.permute_dims(gate_proj_weights)
                )
                up: R.Tensor([batch_size, intermediate_size]) = R.matmul(
                    x, R.permute_dims(up_proj_weights)
                )
                down: R.Tensor([batch_size, hidden_size]) = R.matmul(
                    R.nn.silu(gate) * up, R.permute_dims(down_proj_weights)
                )
                R.output(down)
            return down

    assert_structural_equal(exported_mod, Expected)


def test_generate_parameters():
    """Weights may be expressions in terms of other parameters

    Optimizations often require preprocessing of the model weights.

    1. Declare the `nn.Module` members that contain the original model
       weights.  These are used to define the parameter names when
       reading from a Pytorch or Safetensors file.

    2. Declare the `nn.Module` members, with the `weight` field
       in terms of the un-optimized weights.  These `nn.Module`
       do not generate any parameters in the Relax function.

    3. Define the `forward` function in terms of the `nn.Module`
       members for the updated weight tensors.

    The exported Relax function accepts the original model parameters,
    computes the pre-processed weights, and then performs computations
    using the pre-processed weights.

    In this example, the `LiftTransformParams` transform is applied
    immediately, splitting the Relax function into a pre-processing
    step and an execution step.  In practice, this transform would be
    applied much later in an optimization pipeline, to allow optimized
    compute kernels to be recognized.  For example, in some cases
    `R.matmul(x, R.permute_dims(weight))` may be computed more
    efficiently than `R.matmul(x, weight_transpose)`.  For this
    reason, we do *not* apply `LiftTransformParams` as part of the
    export from `nn.Module` to Relax.

    """

    class LlamaMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            # The nn.Linear for the original parameters are present in
            # the model definition, and are still found when
            # collecting a function's parameters.
            self.gate_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.down_proj = nn.Linear(
                intermediate_size,
                hidden_size,
                dtype="float16",
                bias=False,
            )

            # At runtime, we'd like to have a single concatenated
            # tensor containing both the gate and up projection
            # weights.  We also want to use it in the `forward`
            # function as if it owned its own weights.
            self.gate_up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )

            # The weight tensor of `gate_up_proj` can be overwritten
            # in terms of the original `gate_proj` and `up_proj`
            # tensors.
            self.gate_up_proj.weight = nn.op.concat(
                [self.gate_proj.weight, self.up_proj.weight], dim=0, name="gate_up_proj_weights"
            )

        def forward(self, x: nn.Tensor):
            # Even though the `gate_up_proj` weights are defined as an
            # expression rather than a `nn.Parameter`, the `forward`
            # function does not require any special handling for it.
            concat_gate_up = self.gate_up_proj(x)
            gate, up = nn.op.split(concat_gate_up, 2, axis=-1)
            return self.down_proj(nn.op.silu(gate) * up)

    hidden_size = 4096
    intermediate_size = 11008
    slm_mod = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor((tir.Var("batch_size", "int64"), hidden_size), "float16")
            },
        },
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            x: R.Tensor(["batch_size", hidden_size], "float16"),
            # The function's parameters are defined by the
            # `nn.Parameter` instances, and still reference the
            # original `gate_proj` and `up_proj` weights.  This
            # maintains compatibility with named model weights in a
            # Pytorch or Safetensors file.
            gate_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            up_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            down_proj_weights: R.Tensor([hidden_size, intermediate_size], "float16"),
        ):
            R.func_attr({"num_input": 1})
            batch_size = T.int64()
            with R.dataflow():
                # At this stage of compilation, the concatenation is
                # written within the body of the function.  This will
                # later be extracted into a pre-processing step using
                # `relax.transform.LiftTransformParams`.
                gate_up_proj_weights: R.Tensor(
                    [intermediate_size * 2, hidden_size], "float16"
                ) = R.concat([gate_proj_weights, up_proj_weights], axis=0)
                gate_up: R.Tensor([batch_size, intermediate_size * 2], "float16") = R.matmul(
                    x, R.permute_dims(gate_up_proj_weights)
                )
                gate_up_split = R.split(gate_up, 2, axis=-1)
                gate = gate_up_split[0]
                up = gate_up_split[1]
                down: R.Tensor([batch_size, hidden_size], "float16") = R.matmul(
                    R.nn.silu(gate) * up, R.permute_dims(down_proj_weights)
                )
                R.output(down)
            return down

    assert_structural_equal(exported_mod, Expected)

    @I.ir_module
    class ExpectedAfterLift:
        @R.function
        def forward(
            x: R.Tensor(["batch_size", hidden_size], "float16"),
            # After `relax.transform.LiftTransformParams`, the
            # `gate_proj` and `up_proj` weights have been concatenated
            # together.
            gate_up_proj_weights_transpose: R.Tensor(
                [hidden_size, intermediate_size * 2], "float16"
            ),
            down_proj_weights_transpose: R.Tensor([intermediate_size, hidden_size], "float16"),
        ):
            R.func_attr({"num_input": 1})
            batch_size = T.int64()
            with R.dataflow():
                gate_up: R.Tensor([batch_size, intermediate_size * 2], "float16") = R.matmul(
                    x, gate_up_proj_weights_transpose
                )
                gate_up_split = R.split(gate_up, 2, axis=-1)
                gate = gate_up_split[0]
                up = gate_up_split[1]
                down: R.Tensor([batch_size, hidden_size], "float16") = R.matmul(
                    R.nn.silu(gate) * up, down_proj_weights_transpose
                )
                R.output(down)
            return down

        @R.function
        def transform_params(
            model_params: R.Tuple(
                R.Tensor([intermediate_size, hidden_size], "float16"),
                R.Tensor([intermediate_size, hidden_size], "float16"),
                R.Tensor([hidden_size, intermediate_size], "float16"),
            )
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gate_proj_weights: R.Tensor(
                    [intermediate_size, hidden_size], "float16"
                ) = model_params[0]
                up_proj_weights: R.Tensor(
                    [intermediate_size, hidden_size], "float16"
                ) = model_params[1]
                gate_up_proj_weights: R.Tensor(
                    [intermediate_size * 2, hidden_size], "float16"
                ) = R.concat([gate_proj_weights, up_proj_weights], axis=0)
                gate_up_proj_weights_transpose: R.Tensor(
                    [hidden_size, intermediate_size * 2], "float16"
                ) = R.permute_dims(gate_up_proj_weights)
                down_proj_weights: R.Tensor(
                    [hidden_size, intermediate_size], "float16"
                ) = model_params[2]
                down_proj_weights_transpose: R.Tensor(
                    [intermediate_size, hidden_size], "float16"
                ) = R.permute_dims(down_proj_weights)
                output = (gate_up_proj_weights_transpose, down_proj_weights_transpose)
                R.output(output)
            return output

    lifted_mod = relax.transform.LiftTransformParams(shared_transform=True)(exported_mod)
    assert_structural_equal(lifted_mod, ExpectedAfterLift)


if __name__ == "__main__":
    tvm.testing.main()
