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

        @R.function
        def main_transform_params() -> R.Tuple(R.Object, R.Object):
            cls = Expected
            lv: R.Object = R.call_packed("get_item", R.prim_value(1), sinfo_args=(R.Object,))
            lv1: R.Object = R.call_packed("set_item", R.prim_value(0), lv, sinfo_args=(R.Object,))
            lv2: R.Tuple = R.vm.kill_object(lv)
            lv1_1: R.Object = R.call_packed("get_item", R.prim_value(0), sinfo_args=(R.Object,))
            lv3 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1_1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            lv4: R.Object = R.call_packed("set_item", R.prim_value(1), lv3, sinfo_args=(R.Object,))
            lv5: R.Tuple = R.vm.kill_object(lv1_1)
            gv: R.Tuple(R.Object, R.Object) = (lv1, lv4)
            return gv

    after = LazyTransformParams()(Before)
    tvm.ir.assert_structural_equal(after, Expected, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
