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
import os
import tempfile

import pytest

import tvm
from tvm.script import ir as I, tir as T, relax as R
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec


def _gen_extern_module(mod_dir, file):
    src = """#include <dlpack/dlpack.h>
    #include <tvm/runtime/packed_func.h>

    int f_matmul(DLTensor* a, DLTensor* b, DLTensor* c) { return 0; }

    TVM_DLL_EXPORT_TYPED_FUNC(matmul, f_matmul)"""
    with open(f"{mod_dir}/{file}.cc", "w") as cc_file:
        cc_file.write(src)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.system(
        f"gcc -c {mod_dir}/{file}.cc "
        f"-o {mod_dir}/{file}.o "
        f"-I{cur_dir}/../../../include "
        f"-I{cur_dir}/../../../3rdparty/dlpack/include "
        f"-I{cur_dir}/../../../3rdparty/dmlc-core/include"
    )
    return f"{mod_dir}/{file}.o"


def test_extern_module():
    shape_a = ("a", "b", "c", "d", 1, 2, 3, 4)
    shape_b = ("c", "d", "e", "f", 5, 6, 7, 8)
    shape_c = ("a", "b", "c", "d", "e", "f", 9, 10)
    dtype = "float32"
    tmp_dir = tempfile.mkdtemp()
    obj_file = _gen_extern_module(tmp_dir, "test")
    func_name = "matmul"
    os.system(f"ls {tmp_dir}")

    ext_mod = nn.ExternModule(
        module_spec=spec.ExternModuleSpec(
            filename=obj_file,
            functions=[
                spec.ExternFunctionSpec(
                    symbol=func_name,
                    args=[
                        spec.Tensor(shape_a, dtype),
                        spec.Tensor(shape_b, dtype),
                    ],
                    ret=spec.Tensor(shape_c, dtype),
                )
            ],
        )
    )

    class MatmulModule(nn.Module):
        def __init__(self) -> None:
            self.Matmul = ext_mod

        def forward(self, a: nn.Tensor, b: nn.Tensor):
            return self.Matmul.get_extern_func(func_name)(a, b)

    matmul_mod = MatmulModule()
    ir_module, _ = matmul_mod.export_tvm(
        spec={
            "forward": {
                "a": spec.Tensor(shape_a, dtype),
                "b": spec.Tensor(shape_b, dtype),
            }
        }
    )

    @R.function
    def forward(
        a_1: R.Tensor(("a", "b", "c", "d", 1, 2, 3, 4), dtype="float32"),
        b_1: R.Tensor(("c", "d", "e", "f", 5, 6, 7, 8), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(
        R.Tensor(("a", "b", "c", "d", "e", "f", 9, 10), dtype="float32"), R.Tuple(R.Object)
    ):
        a = T.int64()
        b = T.int64()
        c = T.int64()
        d = T.int64()
        e = T.int64()
        f = T.int64()
        with R.dataflow():
            matmul = R.call_dps_packed(
                "matmul",
                (a_1, b_1),
                out_sinfo=R.Tensor((a, b, c, d, e, f, 9, 10), dtype="float32"),
            )
            gv1: R.Tuple(
                R.Tensor((a, b, c, d, e, f, 9, 10), dtype="float32"), R.Tuple(R.Object)
            ) = matmul, (_io,)
            R.output(gv1)
        return gv1

    tvm.ir.assert_structural_equal(ir_module["forward"], forward)


if __name__ == "__main__":
    tvm.testing.main()
