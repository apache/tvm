import pytest
import tvm

from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm.tir import Schedule


# fmt: off
@tvm.script.ir_module
class Conv2dInt8:
    @T.prim_func
    def main(p0: T.Buffer[(16, 14, 14, 256), "int8"], p1: T.Buffer[(1024, 1, 1, 256), "int8"], p2: T.Buffer[(1, 1, 1, 1024), "int32"], p3: T.Buffer[(1, 1, 1, 1024), "int32"], p4: T.Buffer[1024, "int32"], p5: T.Buffer[1024, "int32"], p6: T.Buffer[1024, "int32"], p7: T.Buffer[1, "int32"], p8: T.Buffer[(16, 14, 14, 1024), "int32"], compute: T.Buffer[(16, 14, 14, 1024), "int32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        compile_engine_const = T.alloc_buffer([], dtype="int32")
        pad_temp = T.alloc_buffer([16, 14, 14, 256], dtype="int8")
        conv2d_nhwc = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        T_subtract = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        T_add = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        compute_1 = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        T_add_1 = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        compute_2 = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        T_subtract_1 = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        compute_3 = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        T_add_2 = T.alloc_buffer([16, 14, 14, 1024], dtype="int32")
        with T.block("compile_engine_const"):
            vi = T.axis.spatial(1, 0)
            T.reads()
            T.writes(compile_engine_const[()])
            compile_engine_const[()] = 59
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 256):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(p0[i0_1, i1_1, i2_1, i3_1])
                T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 14, 14, 1024, 1, 1, 256):
            with T.block("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                T.writes(conv2d_nhwc[nn, yy, xx, ff])
                with T.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], "int32") * T.cast(p1[ff, ry, rx, rc], "int32")
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 1024):
            with T.block("T_subtract"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                T.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 1024):
            with T.block("T_add"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                T.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 14, 14, 1024):
            with T.block("compute"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2])
                T.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.q_multiply_shift_per_axis(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2], 31, False, True, dtype="int32")
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 14, 14, 1024):
            with T.block("T_add_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                T.reads(compile_engine_const[()], compute_1[ax0, ax1, ax2, ax3])
                T.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = compile_engine_const[()] + compute_1[ax0, ax1, ax2, ax3]
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 14, 14, 1024):
            with T.block("compute_1"):
                i0_5, i1_5, i2_5, i3_5 = T.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                T.reads(T_add_1[i0_5, i1_5, i2_5, i3_5])
                T.writes(compute_2[i0_5, i1_5, i2_5, i3_5])
                compute_2[i0_5, i1_5, i2_5, i3_5] = T.max(T.min(T_add_1[i0_5, i1_5, i2_5, i3_5], 255), 0)
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 14, 14, 1024):
            with T.block("T_subtract_1"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                T.reads(compute_2[ax0, ax1, ax2, ax3], p7[0])
                T.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] - p7[0]
        for i0_7, i1_7, i2_7, i3_7 in T.grid(16, 14, 14, 1024):
            with T.block("compute_2"):
                i0_8, i1_8, i2_8, i3_8 = T.axis.remap("SSSS", [i0_7, i1_7, i2_7, i3_7])
                T.reads(T_subtract_1[i0_8, i1_8, i2_8, i3_8])
                T.writes(compute_3[i0_8, i1_8, i2_8, i3_8])
                compute_3[i0_8, i1_8, i2_8, i3_8] = T.q_multiply_shift(T_subtract_1[i0_8, i1_8, i2_8, i3_8], 1408572815, 31, 1, dtype="int32")
        for i0_9, i1_9, i2_9, i3_9 in T.grid(16, 14, 14, 1024):
            with T.block("T_add_2"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0_9, i1_9, i2_9, i3_9])
                T.reads(compute_3[ax0, ax1, ax2, ax3], p8[ax0, ax1, ax2, ax3])
                T.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = compute_3[ax0, ax1, ax2, ax3] + p8[ax0, ax1, ax2, ax3]
        for i0_10, i1_10, i2_10, i3_10 in T.grid(16, 14, 14, 1024):
            with T.block("compute_3"):
                i0_11, i1_11, i2_11, i3_11 = T.axis.remap("SSSS", [i0_10, i1_10, i2_10, i3_10])
                T.reads(T_add_2[i0_11, i1_11, i2_11, i3_11])
                T.writes(compute[i0_11, i1_11, i2_11, i3_11])
                compute[i0_11, i1_11, i2_11, i3_11] = T.max(T.min(T_add_2[i0_11, i1_11, i2_11, i3_11], 255), 0)

# fmt: on


def test_conv2d_int8():
    sch = Schedule(Conv2dInt8)

    conv2d = sch.get_block("conv2d_nhwc")
    sch.cache_write(conv2d, 0, "shared")

    with pytest.raises(tvm.tir.ScheduleError) as e:
        sch.reverse_compute_inline(sch.get_block("T_add_1"))

    err_msg = "The block is only allowed to read a single buffer region, but it reads 2 region(s)"
    assert err_msg in str(e)

    ms.schedule_rule.InlineConstantScalars().apply(sch, sch.get_block("compile_engine_const"))
    sch.reverse_compute_inline(sch.get_block("T_add_1"))


if __name__ == "__main__":
    tvm.testing.main()
