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

import tvm
import tvm.testing

from tvm import relax
from tvm.script import relax as R, tir as T
from tvm.script import ir as I
from tvm.relax.transform import LazyTransformParams


def test_lazy_transform_params():
    @I.ir_module
    class Before:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            )
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv2)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ):
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            cls = Expected
            lv: R.Object = R.call_packed("get_item", R.prim_value(1), sinfo_args=(R.Object,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                lv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv_m: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            _: R.Object = R.call_packed("set_item", R.prim_value(0), lv_m, sinfo_args=(R.Object,))
            _1: R.Tuple = R.vm.kill_object(lv_m)
            lv1: R.Object = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
            gv3: R.Tensor((3, 16, 3, 3), dtype="float32") = R.match_cast(
                lv1, R.Tensor((3, 16, 3, 3), dtype="float32")
            )
            lv1_m: R.Tensor((3, 16, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1_m,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            _2: R.Tuple = R.vm.kill_object(lv1_m)
            _3: R.Object = R.call_packed("set_item", R.prim_value(1), lv2, sinfo_args=(R.Object,))
            gv: R.Tuple = R.tuple()
            return gv

    after = LazyTransformParams()(Before)
    tvm.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_get_item_only():
    @I.ir_module
    class Before:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            )
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            lv3 = R.add(lv2, R.const(1, "float32"))
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv3)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ):
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params() -> (
            R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
            )
        ):
            cls = Expected
            gv: R.Object = R.call_packed("get_item_0", R.prim_value(1), sinfo_args=(R.Object,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                gv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            gv2: R.Object = R.call_packed("get_item_0", R.prim_value(0), sinfo_args=(R.Object,))
            gv3: R.Tensor((3, 16, 3, 3), dtype="float32") = R.match_cast(
                gv2, R.Tensor((3, 16, 3, 3), dtype="float32")
            )
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            lv3: R.Tensor((16, 3, 3, 3), dtype="float32") = R.add(lv2, R.const(1, "float32"))
            gv_1: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
            ) = (lv, lv3)
            return gv_1

    after = LazyTransformParams(fget_item="get_item_0", fset_item=None)(Before)
    tvm.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_extra_params():
    @I.ir_module
    class Before:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
            )
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            lv3 = R.add(lv2, R.const(1, "float32"))
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv3)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(
            w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
        ):
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params(loader: R.Object) -> R.Tuple:
            cls = Expected
            gv: R.Object = R.call_packed(
                "get_item", loader, R.prim_value(1), sinfo_args=(R.Object,)
            )
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                gv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            _: R.Object = R.call_packed("set_item", R.prim_value(0), lv, sinfo_args=(R.Object,))
            _1: R.Tuple = R.vm.kill_object(lv)
            gv2: R.Object = R.call_packed(
                "get_item", loader, R.prim_value(0), sinfo_args=(R.Object,)
            )
            gv3: R.Tensor((3, 16, 3, 3), dtype="float32") = R.match_cast(
                gv2, R.Tensor((3, 16, 3, 3), dtype="float32")
            )
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            _2: R.Tuple = R.vm.kill_object(lv1)
            lv3: R.Tensor((16, 3, 3, 3), dtype="float32") = R.add(lv2, R.const(1, "float32"))
            _3: R.Object = R.call_packed("set_item", R.prim_value(1), lv3, sinfo_args=(R.Object,))
            gv_1: R.Tuple = R.tuple()
            return gv_1

    after = LazyTransformParams(
        extra_get_item_params=[relax.Var("loader", relax.ObjectStructInfo())]
    )(Before)
    tvm.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_lazy_transform_params_with_symbolic_vars():
    @I.ir_module
    class Before:
        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((16, 16), dtype="float32"),
                R.Shape(
                    ["slice_index"],
                ),
            ),
        ):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before

            slice_index = T.int64()

            param = params[0]
            transformed = R.call_tir(
                cls.slice_buffer,
                (param,),
                tir_vars=[slice_index],
                out_sinfo=R.Tensor((16,), dtype="float32"),
            )
            output = (transformed,)
            return output

        @T.prim_func(private=True)
        def slice_buffer(
            Input: T.Buffer((16, 16), "float32"),
            slice_index: T.int64,
            Output: T.Buffer(16, "float32"),
        ):
            for i in T.grid(16):
                with T.block("slice_buffer"):
                    vi = T.axis.remap("S", [i])
                    Output[vi] = Input[slice_index, vi]

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main_transform_params(slice_shape_expr: R.Shape(["slice_index"])):
            cls = Expected

            slice_index = T.int64()

            param = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
            gv: R.Tensor((16, 16), dtype="float32") = R.match_cast(
                param, R.Tensor((16, 16), dtype="float32")
            )
            param_m: R.Tensor((16, 16), dtype="float32") = gv
            transformed = R.call_tir(
                cls.slice_buffer,
                (param_m,),
                tir_vars=[slice_index],
                out_sinfo=R.Tensor((16,), dtype="float32"),
            )
            unused_1_ = R.vm.kill_object(param_m)
            unused_2_ = R.call_packed(
                "set_item", R.prim_value(0), transformed, sinfo_args=(R.Object,)
            )

            output = R.tuple()
            return output

        @T.prim_func(private=True)
        def slice_buffer(
            Input: T.Buffer((16, 16), "float32"),
            slice_index: T.int64,
            Output: T.Buffer(16, "float32"),
        ):
            for i in T.grid(16):
                with T.block("slice_buffer"):
                    vi = T.axis.remap("S", [i])
                    Output[vi] = Input[slice_index, vi]

    after = LazyTransformParams()(Before)
    tvm.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_param_shape_symbolic():
    @I.ir_module
    class Before:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(var_w1: T.handle, var_out: T.handle):
            ic = T.int32()
            w1 = T.match_buffer(var_w1, (ic, 16, 3, 3), "float32")
            out = T.match_buffer(var_out, (16, ic, 3, 3), "float32")
            for ax0, ax1, ax2, ax3 in T.grid(16, ic, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((3, "ic", 3, 3), dtype="float32"),
                R.Tensor((16, 16, 3, 3), dtype="float32"),
            )
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor(("ic", 3, 3, 3), dtype="float32")
        ):
            ic = T.int64()
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Before
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, ic, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((ic, 3, 3, 3), dtype="float32"),
            )
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((ic, 3, 3, 3), dtype="float32"),
            ) = (lv, lv2)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def transform_layout_IOHW_to_OIHW(var_w1: T.handle, var_out: T.handle):
            ic = T.int32()
            w1 = T.match_buffer(var_w1, (ic, 16, 3, 3), "float32")
            out = T.match_buffer(var_out, (16, ic, 3, 3), "float32")
            for ax0, ax1, ax2, ax3 in T.grid(16, ic, 3, 3):
                with T.block("layout_transform"):
                    o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(w1[i, o, h, w])
                    T.writes(out[o, i, h, w])
                    out[o, i, h, w] = w1[i, o, h, w]

        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            ic = T.int64()
            cls = Expected
            gv: R.Object = R.call_packed("get_item", R.prim_value(1), sinfo_args=(R.Object,))
            gv1: R.Tensor((16, 16, 3, 3), dtype="float32") = R.match_cast(
                gv, R.Tensor((16, 16, 3, 3), dtype="float32")
            )
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = gv1
            _: R.Object = R.call_packed("set_item", R.prim_value(0), lv, sinfo_args=(R.Object,))
            _1: R.Tuple = R.vm.kill_object(lv)
            gv2: R.Object = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
            gv3: R.Tensor((3, ic, 3, 3), dtype="float32") = R.match_cast(
                gv2, R.Tensor((3, ic, 3, 3), dtype="float32")
            )
            lv1: R.Tensor((3, ic, 3, 3), dtype="float32") = gv3
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((ic, 3, 3, 3), dtype="float32"),
            )
            _2: R.Tuple = R.vm.kill_object(lv1)
            _3: R.Object = R.call_packed("set_item", R.prim_value(1), lv2, sinfo_args=(R.Object,))
            gv4: R.Tuple = R.tuple()
            return gv4

    after = LazyTransformParams()(Before)
    tvm.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_output_with_use_site():
    @I.ir_module
    class Module:
        @T.prim_func
        def copy(x: T.Buffer((), "float32"), y: T.Buffer((), "float32")):
            with T.block("block"):
                T.reads(x[()])
                T.writes(y[()])
                y[()] = x[()]

        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor((), dtype="float32"))
        ) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")):
            # we expect ToNonDataflow and RemovePurityTracking to be invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Module
            x: R.Tensor((), dtype="float32") = params[0]
            y = R.call_tir(cls.copy, (x,), out_sinfo=R.Tensor((), dtype="float32"))
            z = R.call_tir(cls.copy, (y,), out_sinfo=R.Tensor((), dtype="float32"))
            gv: R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")) = (y, z)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def copy(x: T.Buffer((), "float32"), y: T.Buffer((), "float32")):
            with T.block("block"):
                T.reads(x[()])
                T.writes(y[()])
                y[()] = x[()]

        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            cls = Expected
            x: R.Object = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
            gv: R.Tensor((), dtype="float32") = R.match_cast(x, R.Tensor((), dtype="float32"))
            x_m: R.Tensor((), dtype="float32") = gv
            y = R.call_tir(cls.copy, (x_m,), out_sinfo=R.Tensor((), dtype="float32"))
            _: R.Tuple = R.vm.kill_object(x_m)
            z = R.call_tir(cls.copy, (y,), out_sinfo=R.Tensor((), dtype="float32"))
            _1: R.Object = R.call_packed("set_item", R.prim_value(0), y, sinfo_args=(R.Object,))
            _2: R.Object = R.call_packed("set_item", R.prim_value(1), z, sinfo_args=(R.Object,))
            gv: R.Tuple = R.tuple()
            return gv

    after = LazyTransformParams()(Module)
    tvm.ir.assert_structural_equal(after, Expected, map_free_vars=True)


def test_output():
    target = "llvm"
    dev = tvm.device(target)

    @I.ir_module
    class TransformModule:
        @R.function
        def transform_params(
            params: R.Tuple(
                R.Tensor((3, "ic", 3, 3), dtype="float32"),
                R.Tensor((16, 16, 3, 3), dtype="float32"),
            )
        ) -> R.Tuple(
            R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor(("ic", 3, 3, 3), dtype="float32")
        ):
            R.func_attr({"relax.force_pure": True})
            param0 = params[0]
            param1 = params[1]
            transformed0 = R.permute_dims(param0, [1, 0, 2, 3])
            transformed = (transformed0, param1)
            return transformed

    mod = TransformModule
    mod = relax.transform.LazyTransformParams()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    built = relax.build(mod, target=target)

    params = [
        np.random.random(size=(3, 64, 3, 3)).astype("float32"),
        np.random.random(size=(16, 16, 3, 3)).astype("float32"),
    ]
    transformed = {}
    expected = [params[0].transpose(1, 0, 2, 3), params[1]]

    @tvm.register_func("get_item", override=True)
    def get_item(i):
        return tvm.nd.array(params[i], dev)

    @tvm.register_func("set_item", override=True)
    def set_item(i, value):
        assert i not in transformed, f"Set item called multiple times for index {i}"
        transformed[i] = value.numpy()

    vm = relax.VirtualMachine(built, dev)
    vm["transform_params"]()

    assert sorted(transformed) == list(range(len(transformed)))
    transformed = [value for i, value in sorted(transformed.items())]
    assert len(transformed) == len(expected)

    for expected_i, transformed_i in zip(expected, transformed):
        tvm.testing.assert_allclose(expected_i, transformed_i)


def test_duplicate_outputs():
    """A tensor may be repeated in the output

    This is something that should be avoided upstream, but is a legal
    parameter transformation, and should produce correct output.
    """

    @I.ir_module
    class Before:
        @R.function
        def main_transform_params(
            params: R.Tuple(R.Tensor([16], dtype="int32"), R.Tensor([16], dtype="int32"))
        ):
            R.func_attr({"relax.force_pure": True})
            param0 = params[0]
            param1 = params[1]
            transformed0 = R.add(param0, R.const(1, "int32"))
            transformed1 = R.add(param1, R.const(2, "int32"))
            output = (transformed0, transformed1, transformed0)
            return output

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main_transform_params() -> R.Tuple:
            gv: R.Object = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
            gv1: R.Tensor((16,), dtype="int32") = R.match_cast(gv, R.Tensor((16,), dtype="int32"))
            param0: R.Tensor((16,), dtype="int32") = gv1

            gv2: R.Object = R.call_packed("get_item", R.prim_value(1), sinfo_args=(R.Object,))
            gv3: R.Tensor((16,), dtype="int32") = R.match_cast(gv2, R.Tensor((16,), dtype="int32"))
            param1: R.Tensor((16,), dtype="int32") = gv3

            transformed0: R.Tensor((16,), dtype="int32") = R.add(param0, R.const(1, "int32"))
            _: R.Tuple = R.vm.kill_object(param0)
            _: R.Object = R.call_packed(
                "set_item", R.prim_value(0), transformed0, sinfo_args=(R.Object,)
            )
            _: R.Object = R.call_packed(
                "set_item", R.prim_value(2), transformed0, sinfo_args=(R.Object,)
            )

            transformed1: R.Tensor((16,), dtype="int32") = R.add(param1, R.const(2, "int32"))
            _ = R.vm.kill_object(param1)
            _ = R.call_packed("set_item", R.prim_value(1), transformed1, sinfo_args=(R.Object,))
            output = R.tuple()
            return output

    after = LazyTransformParams()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
