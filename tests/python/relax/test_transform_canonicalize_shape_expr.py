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
from tvm import relax, tirx
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


@I.ir_module
class Before:
    @R.function
    def main(x: R.Tensor(("x_0", "x_1", "x_2", "x_3"), "float32")):
        R.func_attr({"relax.force_pure": True})
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
                if not isinstance(dim, (tirx.IntImm, tirx.Var)):
                    composite_dims.append(dim)

    relax.analysis.post_order_visit(mod["main"], _visit)
    assert not composite_dims


def test_canonicalize_shape_expr_unblocks_vm_shape_lower():
    mod = Before
    mod = relax.transform.CanonicalizeShapeExpr()(mod)
    mod = relax.transform.ComputePrimValue()(mod)
    mod = relax.transform.VMShapeLower()(mod)

    assert any("compute_symbolic_expr" in gv.name_hint for gv in mod.get_global_vars())


@I.ir_module
class ParamCompoundShape:
    @R.function
    def main(x: R.Tensor(("A", "B", "A + B"), "float32")) -> R.Tensor((1,), "float32"):
        out: R.Tensor((1,), "float32") = R.zeros(R.shape([1]), dtype="float32")
        return out


def test_canonicalize_shape_expr_skips_parameter_struct_info():
    mod = relax.transform.CanonicalizeShapeExpr()(ParamCompoundShape)
    param_shape = mod["main"].params[0].struct_info.shape

    assert any(not isinstance(dim, (tirx.IntImm, tirx.Var)) for dim in param_shape.values)


@I.ir_module
class NestedParamCompound:
    @R.function
    def main(x: R.Tensor(("n",), "float32")):
        @R.function
        def inner(y: R.Tensor(("a", "b", "a + b"), "float32")) -> R.Tensor((1,), "float32"):
            out: R.Tensor((1,), "float32") = R.zeros(R.shape([1]), dtype="float32")
            return out

    return inner


def test_canonicalize_shape_expr_nested_function_no_block_pollution():
    mod = relax.transform.CanonicalizeShapeExpr()(NestedParamCompound)

    leaked = [
        binding
        for block in mod["main"].body.blocks
        for binding in block.bindings
        if isinstance(binding, relax.MatchCast)
    ]

    assert not leaked


if __name__ == "__main__":
    tvm.testing.main()
