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
import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

SOURCE_CODE = """
#include <dlpack/dlpack.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/data_type.h>

namespace {

int _scalar_add(DLTensor* a, DLTensor* b, DLTensor* c) {
  using namespace tvm::runtime;
  ICHECK(a->ndim == 0);
  ICHECK(b->ndim == 0);
  ICHECK(c->ndim == 0);
  ICHECK(DataType(a->dtype) == DataType::Float(32));
  ICHECK(DataType(b->dtype) == DataType::Float(32));
  ICHECK(DataType(c->dtype) == DataType::Float(32));
  float* a_data = static_cast<float*>(a->data);
  float* b_data = static_cast<float*>(b->data);
  float* c_data = static_cast<float*>(c->data);
  *c_data = *a_data + *b_data;
  return 0;
}

int _test_sym(DLTensor* a, DLTensor* b, DLTensor* c) {
  using namespace tvm::runtime;
  ICHECK(a->ndim == 3);
  ICHECK(b->ndim == 3);
  ICHECK(c->ndim == 4);
  ICHECK(DataType(a->dtype) == DataType::Float(32));
  ICHECK(DataType(b->dtype) == DataType::Float(32));
  ICHECK(DataType(c->dtype) == DataType::Float(32));
  int x = a->shape[0];
  int y = a->shape[1];
  int z = b->shape[1];
  ICHECK(a->shape[0] == x);
  ICHECK(a->shape[1] == y);
  ICHECK(a->shape[2] == 1);
  ICHECK(b->shape[0] == y);
  ICHECK(b->shape[1] == z);
  ICHECK(b->shape[2] == 5);
  ICHECK(c->shape[0] == x);
  ICHECK(c->shape[1] == y);
  ICHECK(c->shape[2] == z);
  ICHECK(c->shape[3] == 9);
  return 0;
}

}
TVM_DLL_EXPORT_TYPED_FUNC(ext_scalar_add, _scalar_add);
TVM_DLL_EXPORT_TYPED_FUNC(ext_test_sym, _test_sym);
"""


def test_extern_module():
    shape_a = ("x", "y", 1)
    shape_b = ("y", "z", 5)
    shape_c = ("x", "y", "z", 9)
    dtype = "float32"

    class MyExtMod(nn.SourceModule):
        def __init__(self):
            super().__init__(
                source_code=SOURCE_CODE,
                source_format="cpp",
                functions={
                    "ext_scalar_add": spec.ExternFunctionSpec(
                        args=[
                            spec.Tensor((), dtype),
                            spec.Tensor((), dtype),
                        ],
                        ret=spec.Tensor((), dtype),
                    ),
                    "ext_test_sym": spec.ExternFunctionSpec(
                        args=[
                            spec.Tensor(shape_a, dtype),
                            spec.Tensor(shape_b, dtype),
                        ],
                        ret=spec.Tensor(shape_c, dtype),
                    ),
                },
                compile_options=None,
                compiler=None,
                output_format="obj",
            )

        def scalar_add(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
            return self.get_extern_func("ext_scalar_add")(a, b)

        def test_sym(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
            return self.get_extern_func("ext_test_sym")(a, b)

    my_ext_mod = MyExtMod()

    class TestModule(nn.Module):
        def __init__(self) -> None:
            self.extern_matmul = my_ext_mod

        def scalar_add(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
            return self.extern_matmul.scalar_add(a, b)

        def test_sym(self, a: nn.Tensor, b: nn.Tensor):  # pylint: disable=invalid-name
            return self.extern_matmul.test_sym(a, b)

    model = TestModule()
    ir_module, _ = model.export_tvm(
        spec={
            "scalar_add": {
                "a": spec.Tensor((), dtype),
                "b": spec.Tensor((), dtype),
            },
            "test_sym": {
                "a": spec.Tensor(shape_a, dtype),
                "b": spec.Tensor(shape_b, dtype),
            },
        }
    )

    @I.ir_module
    class ExpectedModule:
        @R.function
        def scalar_add(
            a: R.Tensor((), dtype="float32"), b: R.Tensor((), dtype="float32")
        ) -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                ext_scalar_add = R.call_dps_packed(
                    "ext_scalar_add", (a, b), out_sinfo=R.Tensor((), dtype="float32")
                )
                gv: R.Tensor((), dtype="float32") = ext_scalar_add
                R.output(gv)
            return gv

        @R.function
        def test_sym(
            a: R.Tensor(("x", "y", 1), dtype="float32"), b: R.Tensor(("y", "z", 5), dtype="float32")
        ) -> R.Tensor(("x", "y", "z", 9), dtype="float32"):
            x, y, z = T.int64(), T.int64(), T.int64()
            R.func_attr({"num_input": 2})
            with R.dataflow():
                ext_test_sym = R.call_dps_packed(
                    "ext_test_sym", (a, b), out_sinfo=R.Tensor((x, y, z, 9), dtype="float32")
                )
                gv1: R.Tensor((x, y, z, 9), dtype="float32") = ext_test_sym
                R.output(gv1)
            return gv1

    tvm.ir.assert_structural_equal(ir_module["scalar_add"], ExpectedModule["scalar_add"])
    tvm.ir.assert_structural_equal(ir_module["test_sym"], ExpectedModule["test_sym"])
    assert len(ir_module.attrs["external_mods"]) == 1
    assert ir_module.attrs["external_mods"][0].type_key == "static_library"

    scalar_a = tvm.nd.array(np.array(1.0, dtype="float32"))
    scalar_b = tvm.nd.array(np.array(3.0, dtype="float32"))

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = temp_dir + "/lib.so"
        relax.build(ir_module, target="llvm").export_library(output_path)
        compiled = tvm.runtime.relax_vm.VirtualMachine(
            tvm.runtime.load_module(output_path),
            device=tvm.cpu(),
        )
        scalar_c = compiled["scalar_add"](scalar_a, scalar_b)


def test_extern_spec():
    class TestModule(nn.Module):
        def __init__(self) -> None:
            self.ext_mod = nn.ExternModule(
                spec.ExternModuleSpec(
                    library=tvm.runtime.Module(None),
                    functions=[
                        spec.ExternFunctionSpec(
                            args=[
                                spec.Tensor((2, 4), "float16"),
                                spec.ConstInt(),
                                spec.ConstInt("int32"),
                                spec.ConstFloat(),
                                spec.ConstFloat("float16"),
                                spec.ConstString(),
                            ],
                            ret=spec.Tensor((2, 4), "float16"),
                            symbol="test",
                        )
                    ],
                )
            )

        def forward(self, x: nn.Tensor):
            return self.ext_mod.get_extern_func("test")(x, 1, 2, 3.0, 4.0, "123")

    @I.ir_module
    class ExpectedModule:
        I.module_attrs({"external_mods": [None]})

        @R.function
        def forward(x: R.Tensor((2, 4), dtype="float16")) -> R.Tensor((2, 4), dtype="float16"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                test = R.call_dps_packed(
                    "test",
                    (
                        x,
                        R.prim_value(1),
                        R.prim_value(T.int32(2)),
                        R.prim_value(T.float32(3)),
                        R.prim_value(T.float16(4)),
                        R.str("123"),
                    ),
                    out_sinfo=R.Tensor((2, 4), dtype="float16"),
                )
                gv: R.Tensor((2, 4), dtype="float16") = test
                R.output(gv)
            return gv

    model = TestModule()
    ir_module, _ = model.export_tvm(
        spec={
            "forward": {
                "x": spec.Tensor((2, 4), "float16"),
            },
        }
    )
    tvm.ir.assert_structural_equal(ir_module, ExpectedModule)


if __name__ == "__main__":
    tvm.testing.main()
