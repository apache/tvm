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
import pytest

import tvm
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.tir import Var
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.relax.testing import get_relax_matmul_module


has_cutlass = tvm.get_global_func("relax.ext.cutlass", True)

cutlass_enabled = pytest.mark.skipif(
    not has_cutlass,
    reason="CUTLASS not enabled.",
)

pytestmark = [cutlass_enabled]


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


@pytest.mark.parametrize(
    "shape_a, shape_b, shape_out, dtype, value_table, var_table",
    [
        (("n", 4, 16), ("n", 16, 8), ("n", 4, 8), "float16", {"n": 16}, {"n": Var("n", "int64")}),
    ],
)
def test_extern_module(shape_a, shape_b, shape_out, dtype, value_table, var_table):
    def shape_converter(shape, table):
        out_shape = []
        for value in shape:
            if isinstance(value, str):
                out_shape.append(table[value])
            else:
                out_shape.append(value)
        return tuple(out_shape)

    Module = get_relax_matmul_module(
        shape_converter(shape_a, var_table),
        shape_converter(shape_b, var_table),
        dtype,
    )

    mod = partition_for_cutlass(Module, True)
    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)

    cutlass_matmul_mod = nn.ExternModule(
        module_spec=spec.ExternModuleSpec(
            filename="./tmp/cutlass.o",
            functions=[
                spec.ExternFunctionSpec(
                    symbol="fused_relax_matmul_cutlass",
                    args=[
                        spec.Tensor(shape_a, dtype),
                        spec.Tensor(shape_b, dtype),
                    ],
                    ret=spec.Tensor(shape_out, dtype),
                )
            ],
        )
    )

    class MatmulModule(nn.Module):
        def __init__(self) -> None:
            self.Matmul = cutlass_matmul_mod

        def forward(self, a: nn.Tensor, b: nn.Tensor):
            return self.Matmul.get_extern_func("fused_relax_matmul_cutlass")(a, b)

    matmul_mod = MatmulModule()
    ir_module, _ = matmul_mod.export_tvm(
        spec={
            "forward": {
                "a": spec.Tensor(shape_a, dtype),
                "b": spec.Tensor(shape_b, dtype),
            }
        }
    )
    ex = relax.build(ir_module, "cuda")

    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)
    f_init = vm["_initialize_effect"]
    f_forward = vm["forward"]

    io_state = f_init()[0]
    a_np = np.random.randn(*shape_converter(shape_a, value_table)).astype(dtype)
    b_np = np.random.randn(*shape_converter(shape_b, value_table)).astype(dtype)
    inputs = [tvm.nd.array(a_np, dev), tvm.nd.array(b_np, dev), io_state]
    out = f_forward(*inputs)[0].numpy()
    ref = np.matmul(a_np, b_np)
    tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
