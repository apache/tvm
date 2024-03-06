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
# pylint: disable=missing-docstring
import torch

import tvm
import tvm.testing
from tvm import tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op, spec
from tvm.runtime import NDArray

from tvm.script import ir as I, relax as R


def test_debug_print():
    class Layer(nn.Module):
        def forward(self, x: nn.Tensor):  # pylint: disable=invalid-name
            op.print_(x)
            return x

    model = Layer().jit(
        spec={
            "forward": {"x": spec.Tensor([10, 5], dtype="float32")},
        },
        debug=True,
    )
    x = torch.rand((10, 5), dtype=torch.float32)  # pylint: disable=invalid-name
    y = model["forward"](x)  # pylint: disable=invalid-name
    assert isinstance(y, torch.Tensor)


def test_debug_print_well_formed():
    class Layer(nn.Module):
        def forward(self, state: nn.Tensor):
            state = state * 2.0
            op.print_(state)
            state = state * 2.0
            return state

    forward_code = Layer.forward.__wrapped__.__code__
    debug_location = f"{forward_code.co_filename}:{forward_code.co_firstlineno+2}"

    model, _ = Layer().export_tvm(
        spec={
            "forward": {"state": spec.Tensor([10, 5], dtype="float32")},
        },
        debug=True,
    )

    @I.ir_module
    class Expected:
        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io = R.null_value()
                gv = (_io,)
                R.output(gv)
            return gv

        @R.function(pure=False)
        def forward(
            state: R.Tensor((10, 5), dtype="float32"), _io: R.Object
        ) -> R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tuple(R.Object)):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                mul = R.multiply(state, R.const(2, "float32"))
                R.output(mul)

            _io1 = R.call_packed(
                "vm.builtin.invoke_debug_func",
                _io,
                R.str("vm.builtin.debug_print"),
                R.str(debug_location),
                mul,
                sinfo_args=(R.Object,),
            )

            with R.dataflow():
                mul1 = R.multiply(mul, R.const(2, "float32"))
                gv1 = mul1, (_io1,)
                R.output(gv1)

            return gv1

    tvm.ir.assert_structural_equal(Expected, model)


def test_debug_func():
    @tvm.register_func("testing.relax.frontend.nn.test_debug_func")
    def _debug(  # pylint: disable=too-many-arguments
        lineno: str,
        tensor: NDArray,
        const_int: int,
        const_float: float,
        const_str: str,
        var_int: int,
    ) -> None:
        assert "test_frontend_nn_debug.py" in lineno
        assert tensor.shape == (10, 5)
        assert const_int == 1
        assert const_float == 2.0
        assert const_str == "test"
        assert var_int == 8

    class Layer(nn.Module):
        def forward(self, x: nn.Tensor, v: tir.Var):  # pylint: disable=invalid-name
            op.debug_func("testing.relax.frontend.nn.test_debug_func", x, 1, 2.0, "test", v)
            return x

    model = Layer().jit(
        spec={
            "forward": {
                "x": spec.Tensor([10, 5], dtype="float32"),
                "v": "int",
            },
        },
        debug=True,
    )
    x = torch.rand((10, 5), dtype=torch.float32)  # pylint: disable=invalid-name
    y = model["forward"](x, 8)  # pylint: disable=invalid-name
    assert isinstance(y, torch.Tensor)


if __name__ == "__main__":
    tvm.testing.main()
