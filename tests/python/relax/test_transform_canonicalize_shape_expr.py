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

import tvm.testing
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@I.ir_module
class Before:
    @R.function
    def main(x: R.Tensor(("x_0", "x_1", "x_2", "x_3"), "float32")):
        x_0, x_1, x_2, x_3 = T.int64(), T.int64(), T.int64(), T.int64()
        out: R.Tensor((T.int64(4) * (x_0 * x_1 * x_2 * x_3),), "float32") = R.zeros(
            R.shape([T.int64(4) * (x_0 * x_1 * x_2 * x_3)]), dtype="float32"
        )
        return out


def test_canonicalize_shape_expr_removes_composite_dims():
    mod = relax.transform.CanonicalizeShapeExpr()(Before)
    composite_dims = []

    def _visit(expr):
        if isinstance(expr, relax.ShapeExpr):
            for dim in expr.values:
                if not isinstance(dim, tir.IntImm | tir.Var):
                    composite_dims.append(dim)

    relax.analysis.post_order_visit(mod["main"], _visit)
    assert not composite_dims


def test_canonicalize_shape_expr_unblocks_vm_shape_lower():
    mod = Before
    mod = relax.transform.CanonicalizeShapeExpr()(mod)
    mod = relax.transform.ComputePrimValue()(mod)
    mod = relax.transform.VMShapeLower()(mod)

    assert any("compute_symbolic_expr" in gv.name_hint for gv in mod.get_global_vars())


if __name__ == "__main__":
    tvm.testing.main()
