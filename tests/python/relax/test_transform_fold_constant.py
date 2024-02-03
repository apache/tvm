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
from tvm.script import ir as I, tir as T, relax as R


def gen_mod(mod, name, binding):
    """Select relax function with name, rename to main and bind constant.

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
        def addone(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")) -> None:
            for i, j in T.grid(16, 16):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            cls = Module
            lv0 = relax.call_tir(cls.addone, (c0,), R.Tensor((16, 16), dtype="float32"))
            return lv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
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
        def func(A: T.Buffer((2, 3), "float32"), B: T.Buffer((3, 2), "float32")) -> None:
            for i, j in T.grid(3, 2):
                with T.block("transpose"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @R.function
        def before(c0: R.Tensor((2, 3), "float32")):
            cls = Module
            lv0 = relax.call_tir(cls.func, (c0,), R.Tensor((3, 2), dtype="float32"))
            return lv0

        @R.function
        def expected(c1: R.Tensor((3, 2), "float32")):
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
        def addone(A: T.Buffer((2, 2), "float32"), B: T.Buffer((2, 2), "float32")) -> None:
            for i, j in T.grid(2, 2):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @R.function
        def before(c0: R.Tensor((2, 2), "float32")):
            cls = Module
            lv0 = relax.call_tir(cls.addone, (c0,), R.Tensor((2, 2), dtype="float32"))
            lv1 = relax.call_tir(cls.addone, (lv0,), R.Tensor((2, 2), dtype="float32"))
            return lv1

        @R.function
        def expected(c1: R.Tensor((2, 2), "float32"), c2: R.Tensor((2, 2), "float32")):
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
        def identity(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")) -> None:
            for i, j in T.grid(16, 16):
                with T.block("identity"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            cls = Module
            with R.dataflow():
                gv0 = relax.call_tir(cls.identity, (c0,), R.Tensor((16, 16), dtype="float32"))
                R.output(gv0)
            return gv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
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
            n = T.int32()
            m = T.int32()
            A = T.match_buffer(a, (n, m))
            B = T.match_buffer(b, (n, m))
            for i, j in T.grid(n, m):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

        @T.prim_func
        def sub(
            A: T.Buffer((16, 16), "float32"),
            B: T.Buffer((16, 16), "float32"),
            C: T.Buffer((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("sub"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] - B[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), x: R.Tensor("float32", ndim=2)):
            n, m = T.int64(), T.int64()
            cls = Module
            x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
            # this line cannot be folded because n is unknown
            lv0 = relax.call_tir(cls.addone, (c0,), R.Tensor((n, 16), dtype="float32"))
            # this line can be folded
            lv1 = relax.call_tir(cls.addone, (c0,), R.Tensor((16, 16), dtype="float32"))
            # this line can be folded because all inputs are const
            lv2 = relax.call_tir(cls.sub, (c0, lv1), R.Tensor((16, 16), dtype="float32"))
            # this line can not be folded because x's shape is unknown
            lv3 = relax.call_tir(cls.sub, (lv2, x), R.Tensor((16, 16), dtype="float32"))
            return (lv0, lv3)

        @R.function
        def expected(
            c0: R.Tensor((16, 16), "float32"),
            c1: R.Tensor((16, 16), "float32"),
            c2: R.Tensor((16, 16), "float32"),
            x: R.Tensor("float32", ndim=2),
        ):
            n, m = T.int64(), T.int64()
            cls = Module
            x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
            # this line cannot be folded because n is unknown
            lv0 = relax.call_tir(cls.addone, (c0,), R.Tensor((n, 16), dtype="float32"))
            # this line can not be folded because x's shape is unknown
            lv3 = relax.call_tir(cls.sub, (c2, x), R.Tensor((16, 16), dtype="float32"))
            return (lv0, lv3)

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
        def addone(A: T.Buffer((16, 16), "int32"), B: T.Buffer((16, 16), "int32")) -> None:
            for i, j in T.grid(16, 16):
                with T.block("addone"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.int32(1)

        @R.function
        def before(c0: R.Tensor((16, 16), "int32")):
            cls = Module
            lv0 = relax.call_tir(cls.addone, (c0,), R.Tensor((16, 16), dtype="int32"))
            return lv0

        @R.function
        def expected(c1: R.Tensor((16, 16), "int32")):
            return c1

    c0_np = np.arange((16 * 16)).astype("int32").reshape(16, 16)
    c1_np = c0_np + 1
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_fold_single_relax_op():
    # put before after in a single module
    @tvm.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                gv = R.add(c0, c0)
                R.output(gv)
            return gv

        @R.function
        def expected(c1: R.Tensor((16, 16), "float32")):
            return c1

    c0_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    c1_np = c0_np + c0_np
    before = gen_mod(Module, "before", {"c0": c0_np})
    expected = gen_mod(Module, "expected", {"c1": c1_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_fold_multiple_relax_ops():
    # put before after in a single module
    @tvm.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), c1: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                lv0 = R.add(c0, c1)
                lv1 = R.multiply(c0, lv0)
                gv = R.subtract(lv1, c1)
                R.output(gv)
            return gv

        @R.function
        def expected(c4: R.Tensor((16, 16), "float32")):
            return c4

    c0_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    c1_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    c2_np = c0_np + c1_np
    c3_np = c0_np * c2_np
    c4_np = c3_np - c1_np
    before = gen_mod(Module, "before", {"c0": c0_np, "c1": c1_np})
    expected = gen_mod(Module, "expected", {"c4": c4_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_do_not_fold_ops_outside_dataflow():
    # put before after in a single module
    @tvm.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            gv = R.add(c0, c0)
            return gv

    c0_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    before = gen_mod(Module, "before", {"c0": c0_np})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, before)


def test_fold_multiple_relax_ops_with_data_dependent_reshape():
    @tvm.script.ir_module
    class Module:
        @R.function
        def before(
            data: R.Tensor((256,), "float32"),
            c0: R.Tensor((2,), "int64"),
            c1: R.Tensor((2,), "int64"),
        ):
            with R.dataflow():
                lv0 = R.add(c0, c0)
                target_shape = R.multiply(lv0, c1)
                lv2: R.Shape(ndim=2) = R.tensor_to_shape(target_shape)
                gv: R.Tensor(ndim=2, dtype="float32") = R.reshape(data, lv2)
                R.output(gv)
            return gv

        @R.function
        def expected(data: R.Tensor((256,), "float32")) -> R.Tensor((16, 16), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((16, 16), dtype="float32") = R.reshape(data, R.shape([16, 16]))
                R.output(gv)
            return gv

    c0_np = [8, 8]
    c1_np = [1, 1]
    before = gen_mod(Module, "before", {"c0": c0_np, "c1": c1_np})
    assert relax.analysis.well_formed(before)

    expected = gen_mod(Module, "expected", {})

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_unsupported_fold_ops_legalized_to_multiple_calls():
    @tvm.script.ir_module
    class Module:
        @R.function
        def before(c0: R.Tensor((16, 16), "float32")):
            with R.dataflow():
                gv = R.nn.relu(c0)
                R.output(gv)
            return gv

    c0_np = np.arange((16 * 16)).astype("float32").reshape(16, 16)
    before = gen_mod(Module, "before", {"c0": c0_np})

    from tvm.relax.transform.legalize_ops.common import register_legalize

    def customized_legalize_relu(bb: relax.BlockBuilder, call: relax.Call):
        from tvm import topi  # pylint: disable=import-outside-toplevel

        x = bb.emit_te(topi.nn.relu, *call.args)
        return bb.call_te(topi.identity, x)

    # register custom legalization for relu that emits multiple bindings for testing
    relu_legalize = tvm.ir.Op.get("relax.nn.relu").get_attr("FLegalize")
    tvm.ir.Op.get("relax.nn.relu").reset_attr("FLegalize")
    register_legalize("relax.nn.relu", customized_legalize_relu)

    after = relax.transform.FoldConstant()(before)
    tvm.ir.assert_structural_equal(after, before)

    # revert to correct legalization of relu
    tvm.ir.Op.get("relax.nn.relu").reset_attr("FLegalize")
    register_legalize("relax.nn.relu", relu_legalize)


def test_fold_shape_computation():
    @I.ir_module
    class Module:
        @R.function
        def before(
            data: R.Tensor((5, 4, 3, 2), dtype="float32"),
            indices: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((1, 1), dtype="int64"):
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(R.shape([5, 4, 3, 2]))
                lv1: R.Tensor((1,), dtype="int64") = R.take(lv, indices, axis=0)
                lv2: R.Tensor((1, 1), dtype="int64") = R.expand_dims(lv1, axis=[0])
                gv: R.Tensor((1, 1), dtype="int64") = R.concat((lv2,), axis=0)
                R.output(gv)
            return gv

        @R.function
        def expected(
            data: R.Tensor((5, 4, 3, 2), dtype="float32"), new_shape: R.Tensor((1, 1), "int64")
        ) -> R.Tensor((1, 1), dtype="int64"):
            return new_shape

    before = gen_mod(Module, "before", {"indices": tvm.nd.array(np.array([0]).astype("int64"))})
    after = relax.transform.FoldConstant()(before)
    np_take = np.take([5, 4, 3, 2], [0], axis=0)
    np_expand = np.expand_dims(np_take, axis=[0])
    np_concat = np.concatenate([np_expand], axis=0)
    expected = gen_mod(Module, "expected", {"new_shape": tvm.nd.array(np_concat)})
    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
