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
import numpy as np

import tvm.script
from tvm.script import tir as T, relax as R


def gen_mod(mod, name, binding):
    """Select relax function with name, rename to main and and bind constant.

    Parameters
    ----------
    mod: IRModule
        The input module

    name: str
        The name of relax function to preserve and rename to main

    binding: Dict[str, array]
        The const parameter bindings
    """
    funcs = {}
    binding = {k: tvm.nd.array(v) for k, v in binding.items()}

    for k, v in mod.functions.items():
        if isinstance(v, tvm.relax.Function):
            if k.name_hint == name:
                # rename to main
                gv = tvm.ir.GlobalVar("main")
                funcs[gv] = tvm.relax.Function(v.params, v.body, v.ret_struct_info).with_attr(
                    "global_symbol", "main"
                )
        else:
            funcs[k] = v
    mod = tvm.IRModule(funcs)
    return relax.transform.BindParams("main", binding)(mod)


def test_one_fold_addone():
    # put before after in a single module
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def addone(A: T.Buffer[(16, 16), "float32"], B: T.Buffer[(16, 16), "float32"]) -> None:
            for i, j in T.grid(16, 16):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            lv0 = relax.call_tir(addone, (c0,), R.Tensor((16, 16), dtype="float32"))
            return lv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
            lv0 = c1
            return c1

    c0_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    c1_np = c0_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_one_fold_transpose():
    # put before after in a single module
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def func(A: T.Buffer[(2, 3), "float32"], B: T.Buffer[(3, 2), "float32"]) -> None:
            for i, j in T.grid(3, 2):
                with T.block("transpose"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @R.function
        def before(c0: R.Tensor((2, 3), "float32")):
            lv0 = relax.call_tir(func, (c0,), R.Tensor((3, 2), dtype="float32"))
            return lv0

        @R.function
        def expected(c1: R.Tensor((3, 2), "float32")):
            lv0 = c1
            return c1

    c0_np = np.arange(2 * 3).astype("float32").reshape(2, 3)
    c1_np = c0_np.T
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_two_hop_addone():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def addone(A: T.Buffer[(2, 2), "float32"], B: T.Buffer[(2, 2), "float32"]) -> None:
            for i, j in T.grid(2, 2):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @R.function
        def before(c0: R.Tensor((2, 2), "float32")):
            lv0 = relax.call_tir(addone, (c0,), R.Tensor((2, 2), dtype="float32"))
            lv1 = relax.call_tir(addone, (lv0,), R.Tensor((2, 2), dtype="float32"))
            return lv1

        @R.function
        def expected(c1: R.Tensor((2, 2), "float32"), c2: R.Tensor((2, 2), "float32")):
            lv0 = c1
            lv1 = c2
            return c2

    c0_np = np.arange((2 * 2)).astype("float32").reshape(2, 2)
    c1_np = c0_np + 1
    c2_np = c1_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np, "c2": c2_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_dataflow_fold():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def identity(A: T.Buffer[(16, 16), "float32"], B: T.Buffer[(16, 16), "float32"]) -> None:
            for i, j in T.grid(16, 16):
                with T.block("identity"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                gv0 = relax.call_tir(identity, (c0,), R.Tensor((16, 16), dtype="float32"))
                R.output(gv0)
            return gv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                gv0 = c1
                R.output(gv0)
            return c1

    c0_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    c1_np = c0_np
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})
    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_fold_mixed_case():
    @tvm.script.ir_module
    class Module:
        # TIR function can handle different cases.
        @T.prim_func
        def addone(a: T.handle, b: T.handle) -> None:
            n = T.var("int32")
            m = T.var("int32")
            A = T.match_buffer(a, (n, m))
            B = T.match_buffer(b, (n, m))
            for i, j in T.grid(n, m):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @T.prim_func
        def sub(
            A: T.Buffer[(16, 16), "float32"],
            B: T.Buffer[(16, 16), "float32"],
            C: T.Buffer[(16, 16), "float32"],
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("sub"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] - B[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), x: R.Tensor("float32", ndim=2)):
            n, m = T.var("int64"), T.var("int64")
            x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
            # this line cannot be folded because n is unknown
            lv0 = relax.call_tir(addone, (c0,), R.Tensor((n, 16), dtype="float32"))
            # this line can be folded
            lv1 = relax.call_tir(addone, (c0,), R.Tensor((16, 16), dtype="float32"))
            # this line can be folded because all inputs are const
            lv2 = relax.call_tir(sub, (c0, lv1), R.Tensor((16, 16), dtype="float32"))
            # this line can not be folded because x's shape is unknown
            lv3 = relax.call_tir(sub, (lv2, x), R.Tensor((16, 16), dtype="float32"))
            return lv3

        @R.function
        def expected(
            c0: R.Tensor((16, 16), "float32"),
            c1: R.Tensor((16, 16), "float32"),
            c2: R.Tensor((16, 16), "float32"),
            x: R.Tensor("float32", ndim=2),
        ) -> R.Tensor:
            n, m = T.var("int64"), T.var("int64")
            x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
            # this line cannot be folded because n is unknown
            lv0 = relax.call_tir(addone, (c0,), R.Tensor((n, 16), dtype="float32"))
            # this line can be folded
            lv1 = c1
            # this line can be folded because all inputs are const
            lv2 = c2
            # this line can not be folded because x's shape is unknown
            lv3 = relax.call_tir(sub, (c2, x), R.Tensor((16, 16), dtype="float32"))
            return lv3

    c0_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    c1_np = c0_np + 1
    c2_np = c0_np - c1_np

    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c0": c0_np, "c1": c1_np, "c2": c2_np})
    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_int32_fold():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def addone(A: T.Buffer[(16, 16), "int32"], B: T.Buffer[(16, 16), "int32"]) -> None:
            for i, j in T.grid(16, 16):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.int32(1)

        @R.function
        def before(c0: R.Tensor((16, 16), "int32")):
            lv0 = relax.call_tir(addone, (c0,), R.Tensor((16, 16), dtype="int32"))
            return lv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "int32")):
            lv0 = c1
            return c1

    c0_np = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    c1_np = c0_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
