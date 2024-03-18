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
from tvm.script import ir as I, relax as R


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


if __name__ == "__main__":
    tvm.testing.main()
