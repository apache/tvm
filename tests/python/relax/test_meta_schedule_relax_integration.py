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
"""Integration test for MetaSchedule"""

import numpy as np
import pytest
import tempfile
import tvm
import tvm.testing
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm import relax, tir
from tvm.ir import transform

from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

# fmt: off
@I.ir_module
class Module0:
    @R.function
    def main(data: R.Tensor((1, 8, 8, 4), dtype="int32")) -> R.Tensor((1, 8, 8, 4), dtype="int32"):
        cls = Module0
        with R.dataflow():
            c = R.const([[[[-171701247],[-1719837685],[1801664104],[-634316588]],[[920159370],[-132073802],[2142531563],[1465185701]],[[-1505608067],[1737948828],[1581089391],[-1986167320]]],[[[-1449581822],[35714587],[496324563],[-1430879015]],[[-1615680873],[1198514997],[1494683955],[1567376558]],[[1319924884],[-380548171],[296785437],[-1546305981]]],[[[-398644701],[-2004794585],[-1850413687],[2072643657]],[[847950121],[-544212073],[-199532669],[-343273682]],[[953721562],[-1930209358],[1573600108],[-577689853]]]], "int32")
            lv: R.Tensor((1, 8, 8, 4), dtype="int32") = R.nn.conv2d(data, c, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=4, data_layout="NHWC", kernel_layout="HWOI", out_layout="NHWC", out_dtype="int32")
            b = R.const([[[[1, 1, 1, 1]]]], "int32")
            lv1: R.Tensor((1, 8, 8, 4), dtype="int32") = R.add(lv, b)
            c1 = R.const([[[[2042349344],[-2076067063],[1528163722],[-1156452837]],[[-2097172051],[1137787079],[-601389657],[1907495997]],[[987801941],[1073738593],[-1410339796],[-689755358]]],[[[90351522],[-44886952],[-1914103775],[-691553659]],[[-1288505112],[-1376578817],[-2067933148],[-1413101824]],[[1261422027],[-156976862],[-1185734459],[1608778622]]],[[[-664209483],[1907479806],[1838595152],[464942526]],[[877953160],[415131837],[-2010736511],[1218242769]],[[-1440127632],[112931],[521745784],[-1931145893]]]], "int32")
            lv2: R.Tensor((1, 8, 8, 4), dtype="int32") = R.nn.conv2d(lv1, c1, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=4, data_layout="NHWC", kernel_layout="HWOI", out_layout="NHWC", out_dtype="int32")
            c2 = R.const([[[[687940110],[-910571705],[-901609800],[-500525928]],[[506872399],[1070176297],[-305936110],[1625439784]],[[-1565626954],[-1705688881],[-866370805],[-1750740826]]],[[[300497007],[-626864803],[390295545],[222549121]],[[319224543],[-2003064970],[657992492],[2014175448]],[[653278589],[-768810984],[-294555581],[-1197167662]]],[[[1703154671],[-1540759805],[-568817430],[-1729755444]],[[-275458074],[2078945571],[1683298006],[-1029327874]],[[1315093181],[159010501],[875694807],[-223655381]]]], "int32")
            lv3: R.Tensor((1, 8, 8, 4), dtype="int32") = R.nn.conv2d(lv2, c2, strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=4, data_layout="NHWC", kernel_layout="HWOI", out_layout="NHWC", out_dtype="int32")
            gv: R.Tensor((1, 8, 8, 4), dtype="int32") = lv3
            R.output(gv)
        return gv

# fmt: on

# fmt: off
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def conv2d(rxplaceholder: T.Buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(4)), "int32"), DepthwiseConv2d: T.Buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(4)), "int32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": True})
        # with T.block("root"):
        PaddedInput = T.alloc_buffer((T.int64(1), T.int64(10), T.int64(10), T.int64(4)), "int32")
        fused_constant = T.allocate_const([-171701247, -1719837685, 1801664104, -634316588, 920159370, -132073802, 2142531563, 1465185701, -1505608067, 1737948828, 1581089391, -1986167320, -1449581822, 35714587, 496324563, -1430879015, -1615680873, 1198514997, 1494683955, 1567376558, 1319924884, -380548171, 296785437, -1546305981, -398644701, -2004794585, -1850413687, 2072643657, 847950121, -544212073, -199532669, -343273682, 953721562, -1930209358, 1573600108, -577689853], "int32", [3, 3, 4, 1])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(10), T.int64(10), T.int64(4)):
            with T.block("PaddedInput"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(rxplaceholder[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
                T.writes(PaddedInput[v_i0, v_i1, v_i2, v_i3])
                PaddedInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i1 and v_i1 < T.int64(9) and T.int64(1) <= v_i2 and v_i2 < T.int64(9), rxplaceholder[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3], 0)
        for b, i, j, c, di, dj in T.grid(T.int64(1), T.int64(8), T.int64(8), T.int64(4), T.int64(3), T.int64(3)):
            with T.block("DepthwiseConv2d"):
                v_b, v_i, v_j, v_c, v_di, v_dj = T.axis.remap("SSSSRR", [b, i, j, c, di, dj])
                fused_constant_1 = T.Buffer((3, 3, 4, 1), "int32", data=fused_constant)
                T.reads(PaddedInput[v_b, v_i + v_di, v_j + v_dj, v_c], fused_constant_1[v_di, v_dj, v_c, T.int64(0)])
                T.writes(DepthwiseConv2d[v_b, v_i, v_j, v_c])
                with T.init():
                    DepthwiseConv2d[v_b, v_i, v_j, v_c] = 0
                DepthwiseConv2d[v_b, v_i, v_j, v_c] = DepthwiseConv2d[v_b, v_i, v_j, v_c] + PaddedInput[v_b, v_i + v_di, v_j + v_dj, v_c] * fused_constant_1[v_di, v_dj, v_c, T.int64(0)]

    @T.prim_func(private=True)
    def conv2d0(rxplaceholder0: T.Buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(4)), "int32"), DepthwiseConv2d0: T.Buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(4)), "int32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": True})
        # with T.block("root"):
        PaddedInput0 = T.alloc_buffer((T.int64(1), T.int64(10), T.int64(10), T.int64(4)), "int32")
        fused_constant0 = T.allocate_const([2042349344, -2076067063, 1528163722, -1156452837, -2097172051, 1137787079, -601389657, 1907495997, 987801941, 1073738593, -1410339796, -689755358, 90351522, -44886952, -1914103775, -691553659, -1288505112, -1376578817, -2067933148, -1413101824, 1261422027, -156976862, -1185734459, 1608778622, -664209483, 1907479806, 1838595152, 464942526, 877953160, 415131837, -2010736511, 1218242769, -1440127632, 112931, 521745784, -1931145893], "int32", [3, 3, 4, 1])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(10), T.int64(10), T.int64(4)):
            with T.block("PaddedInput"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(rxplaceholder0[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
                T.writes(PaddedInput0[v_i0, v_i1, v_i2, v_i3])
                PaddedInput0[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i1 and v_i1 < T.int64(9) and T.int64(1) <= v_i2 and v_i2 < T.int64(9), rxplaceholder0[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3], 0)
        for b, i, j, c, di, dj in T.grid(T.int64(1), T.int64(8), T.int64(8), T.int64(4), T.int64(3), T.int64(3)):
            with T.block("DepthwiseConv2d"):
                v_b, v_i, v_j, v_c, v_di, v_dj = T.axis.remap("SSSSRR", [b, i, j, c, di, dj])
                fused_constant0_1 = T.Buffer((3, 3, 4, 1), "int32", data=fused_constant0)
                T.reads(PaddedInput0[v_b, v_i + v_di, v_j + v_dj, v_c], fused_constant0_1[v_di, v_dj, v_c, T.int64(0)])
                T.writes(DepthwiseConv2d0[v_b, v_i, v_j, v_c])
                with T.init():
                    DepthwiseConv2d0[v_b, v_i, v_j, v_c] = 0
                DepthwiseConv2d0[v_b, v_i, v_j, v_c] = DepthwiseConv2d0[v_b, v_i, v_j, v_c] + PaddedInput0[v_b, v_i + v_di, v_j + v_dj, v_c] * fused_constant0_1[v_di, v_dj, v_c, T.int64(0)]

    @T.prim_func(private=True)
    def fused_conv2d_add(data: T.Buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(4)), "int32"), T_add: T.Buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(4)), "int32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        PaddedInput = T.alloc_buffer((T.int64(1), T.int64(10), T.int64(10), T.int64(4)), "int32")
        DepthwiseConv2d = T.alloc_buffer((T.int64(1), T.int64(8), T.int64(8), T.int64(4)), "int32")
        fused_nn_conv2d_constant = T.allocate_const([1, 1, 1, 1], "int32", [1, 1, 1, 4])
        fused_constant_2 = T.allocate_const([687940110, -910571705, -901609800, -500525928, 506872399, 1070176297, -305936110, 1625439784, -1565626954, -1705688881, -866370805, -1750740826, 300497007, -626864803, 390295545, 222549121, 319224543, -2003064970, 657992492, 2014175448, 653278589, -768810984, -294555581, -1197167662, 1703154671, -1540759805, -568817430, -1729755444, -275458074, 2078945571, 1683298006, -1029327874, 1315093181, 159010501, 875694807, -223655381], "int32", [3, 3, 4, 1])
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(10), T.int64(10), T.int64(4)):
            with T.block("PaddedInput"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(data[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
                T.writes(PaddedInput[v_i0, v_i1, v_i2, v_i3])
                PaddedInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i1 and v_i1 < T.int64(9) and T.int64(1) <= v_i2 and v_i2 < T.int64(9), data[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3], 0)
        for b, i, j, c, di, dj in T.grid(T.int64(1), T.int64(8), T.int64(8), T.int64(4), T.int64(3), T.int64(3)):
            with T.block("DepthwiseConv2d"):
                v_b, v_i, v_j, v_c, v_di, v_dj = T.axis.remap("SSSSRR", [b, i, j, c, di, dj])
                fused_constant_2_1 = T.Buffer((3, 3, 4, 1), "int32", data=fused_constant_2)
                T.reads(PaddedInput[v_b, v_i + v_di, v_j + v_dj, v_c], fused_constant_2_1[v_di, v_dj, v_c, T.int64(0)])
                T.writes(DepthwiseConv2d[v_b, v_i, v_j, v_c])
                with T.init():
                    DepthwiseConv2d[v_b, v_i, v_j, v_c] = 0
                DepthwiseConv2d[v_b, v_i, v_j, v_c] = DepthwiseConv2d[v_b, v_i, v_j, v_c] + PaddedInput[v_b, v_i + v_di, v_j + v_dj, v_c] * fused_constant_2_1[v_di, v_dj, v_c, T.int64(0)]
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(8), T.int64(8), T.int64(4)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                fused_nn_conv2d_constant_1 = T.Buffer((1, 1, 1, 4), "int32", data=fused_nn_conv2d_constant)
                T.reads(DepthwiseConv2d[v_ax0, v_ax1, v_ax2, v_ax3], fused_nn_conv2d_constant_1[v_ax0, T.int64(0), T.int64(0), v_ax3])
                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                T_add[v_ax0, v_ax1, v_ax2, v_ax3] = DepthwiseConv2d[v_ax0, v_ax1, v_ax2, v_ax3] + fused_nn_conv2d_constant_1[v_ax0, T.int64(0), T.int64(0), v_ax3]

    @R.function
    def main(data: R.Tensor((1, 8, 8, 4), dtype="int32")) -> R.Tensor((1, 8, 8, 4), dtype="int32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.fused_conv2d_add, data, out_sinfo=R.Tensor((1, 8, 8, 4), dtype="int32"))
            lv2 = R.call_tir(cls.conv2d, lv, out_sinfo=R.Tensor((1, 8, 8, 4), dtype="int32"))
            lv3 = R.call_tir(cls.conv2d0, lv2, out_sinfo=R.Tensor((1, 8, 8, 4), dtype="int32"))
            gv: R.Tensor((1, 8, 8, 4), dtype="int32") = lv3
            R.output(gv)
        return gv
# fmt: on


def test_extracting_tasks():
    target = "llvm -mcpu=core-avx2 -num-cores=1"

    relax_mod = Module0
    relax_mod = relax.transform.LegalizeOps()(relax_mod)
    relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
    relax_mod = relax.transform.FuseOps()(relax_mod)
    relax_mod = relax.transform.FoldConstant()(relax_mod)
    relax_mod = relax.transform.FuseTIR()(relax_mod)

    relax_expectation = {
        "structural": 2,  # The relax constants do not reach the tir at the lowering.
        "ignore-ndarray": 2,
        "anchor-block": 1,
    }
    for module_equality, count in relax_expectation.items():
        extracted_tasks = ms.relax_integration.extract_tasks(
            relax_mod,
            target,
            {},
            module_equality=module_equality,
        )
        assert len(extracted_tasks) == count

    tir_relax_mod = Module
    tir_relax_expectation = {"structural": 3, "ignore-ndarray": 2, "anchor-block": 1}
    for module_equality, count in tir_relax_expectation.items():
        extracted_tasks = ms.relax_integration.extract_tasks(
            tir_relax_mod,
            target,
            {},
            module_equality=module_equality,
        )
        assert len(extracted_tasks) == count


@pytest.mark.parametrize("module_equality", ["structural", "ignore-ndarray", "anchor-block"])
def test_using_anchor_trace(module_equality):
    relax_mod = Module
    target = "llvm -mcpu=core-avx2 -num-cores=1"

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relax_integration.tune_relax(
            mod=relax_mod,
            params={},
            target=target,
            work_dir=work_dir,
            # for faster tuning
            max_trials_global=100,
            max_trials_per_task=4,
            num_trials_per_iter=4,
            strategy="replay-trace",
            module_equality=module_equality,
            seed=0,
        )

    ms.relax_integration.compile_relax(
        database,
        mod=relax_mod,
        target=target,
        params={},
    )


if __name__ == "__main__":
    tvm.testing.main()
