"""VTA related intrinsics"""
from __future__ import absolute_import as _abs

import tvm
from . import hw_spec as spec
from .runtime import VTA_AXIS, VTA_PUSH_UOP, get_task_qid
from .runtime import SCOPE_OUT, SCOPE_INP, SCOPE_WGT

# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % SCOPE_INP)
def mem_info_inp_buffer():
    return tvm.make.node("MemoryInfo",
                         unit_bits=spec.VTA_INP_ELEM_BYTES * 8,
                         max_simd_bits=spec.VTA_INP_ELEM_BYTES * 8,
                         max_num_bits=spec.VTA_INP_BUFF_SIZE * 8,
                         head_address=None)

@tvm.register_func("tvm.info.mem.%s" % SCOPE_WGT)
def mem_info_wgt_buffer():
    return tvm.make.node("MemoryInfo",
                         unit_bits=spec.VTA_WGT_ELEM_BYTES * 8,
                         max_simd_bits=spec.VTA_WGT_ELEM_BYTES * 8,
                         max_num_bits=spec.VTA_WGT_BUFF_SIZE * 8,
                         head_address=None)

@tvm.register_func("tvm.info.mem.%s" % SCOPE_OUT)
def mem_info_out_buffer():
    return tvm.make.node("MemoryInfo",
                         unit_bits=spec.VTA_OUT_ELEM_BYTES * 8,
                         max_simd_bits=spec.VTA_OUT_ELEM_BYTES * 8,
                         max_num_bits=spec.VTA_OUT_BUFF_SIZE * 8,
                         head_address=None)

def intrin_gevm(mock=False):
    """Vector-matrix multiply intrinsic"""
    wgt_lanes = spec.VTA_WGT_ELEM_BYTES * 8 // spec.VTA_WGT_WIDTH
    assert wgt_lanes == spec.VTA_BLOCK_OUT * spec.VTA_BLOCK_IN
    wgt_shape = (spec.VTA_BLOCK_OUT, spec.VTA_BLOCK_IN)
    assert wgt_shape[0] * wgt_shape[1] == wgt_lanes
    inp_lanes = spec.VTA_INP_ELEM_BYTES * 8 // spec.VTA_INP_WIDTH
    out_lanes = spec.VTA_OUT_ELEM_BYTES * 8 // spec.VTA_OUT_WIDTH
    wgt = tvm.placeholder((wgt_shape[0], wgt_shape[1]),
                          dtype="int%d" % spec.VTA_WGT_WIDTH,
                          name=SCOPE_WGT)
    inp = tvm.placeholder((wgt_shape[1], ),
                          dtype="int%d" % spec.VTA_INP_WIDTH,
                          name=SCOPE_INP)
    k = tvm.reduce_axis((0, wgt_shape[1]), name="k")
    out_dtype = "int%d" % spec.VTA_OUT_WIDTH
    out = tvm.compute((wgt_shape[0],),
                      lambda i: tvm.sum(inp[k].astype(out_dtype) *
                                        wgt[i, k].astype(out_dtype),
                                        axis=[k]),
                      name="out")
    wgt_layout = tvm.decl_buffer(
        wgt.shape, wgt.dtype, SCOPE_WGT,
        scope=SCOPE_WGT, offset_factor=wgt_lanes, data_alignment=wgt_lanes)
    inp_layout = tvm.decl_buffer(
        inp.shape, inp.dtype, SCOPE_INP,
        scope=SCOPE_INP, offset_factor=inp_lanes, data_alignment=inp_lanes)
    out_layout = tvm.decl_buffer(
        out.shape, out.dtype, SCOPE_OUT,
        scope=SCOPE_OUT, offset_factor=out_lanes, data_alignment=out_lanes)

    def intrin_func(ins, outs):
        """Vector-matrix multiply intrinsic function"""
        dinp, dwgt = ins
        dout = outs[0]
        def instr(index):
            """Generate vector-matrix multiply VTA instruction"""
            irb = tvm.ir_builder.create()
            irb.scope_attr(VTA_AXIS, "coproc_scope", get_task_qid(spec.VTA_QID_COMPUTE))
            irb.scope_attr(VTA_AXIS, "coproc_uop_scope", VTA_PUSH_UOP)
            if index == 0 or index == 2:
                irb.emit(tvm.call_extern(
                    "int32", "VTAUopPush",
                    0, 0,
                    dout.access_ptr("rw", "int32"),
                    dinp.access_ptr("r", "int32"),
                    dwgt.access_ptr("r", "int32"),
                    0, 0, 0))
            else:
                irb.emit(tvm.call_extern(
                    "int32", "VTAUopPush",
                    0, 1,
                    dout.access_ptr("rw", "int32"),
                    0,
                    0,
                    0, 0, 0))
            return irb.get()
        # return a triple of normal-set, reset, update
        nop = tvm.make.Evaluate(0)
        if mock:
            return (nop, nop, nop)
        return (instr(0), instr(1), instr(2))

    return tvm.decl_tensor_intrin(out.op, intrin_func,
                                  name="GEVM",
                                  binds={inp: inp_layout,
                                         wgt: wgt_layout,
                                         out: out_layout})


def intrin_gemm(mock=False):
    """Matrix-matrix multiply intrinsic"""
    wgt_lanes = spec.VTA_WGT_ELEM_BYTES * 8 // spec.VTA_WGT_WIDTH
    assert wgt_lanes == spec.VTA_BLOCK_OUT * spec.VTA_BLOCK_IN
    wgt_shape = (spec.VTA_BLOCK_OUT, spec.VTA_BLOCK_IN)
    assert wgt_shape[0] * wgt_shape[1] == wgt_lanes

    inp_lanes = spec.VTA_INP_ELEM_BYTES * 8 // spec.VTA_INP_WIDTH
    assert inp_lanes == spec.VTA_BATCH * spec.VTA_BLOCK_IN
    inp_shape = (spec.VTA_BATCH, spec.VTA_BLOCK_IN)
    assert inp_shape[0] * inp_shape[1] == inp_lanes

    out_lanes = spec.VTA_OUT_ELEM_BYTES * 8 // spec.VTA_OUT_WIDTH
    assert out_lanes == spec.VTA_BATCH * spec.VTA_BLOCK_OUT
    out_shape = (spec.VTA_BATCH, spec.VTA_BLOCK_OUT)
    assert out_shape[0] * out_shape[1] == out_lanes

    wgt = tvm.placeholder((wgt_shape[0], wgt_shape[1]),
                          dtype="int%d" % spec.VTA_WGT_WIDTH,
                          name=SCOPE_WGT)
    inp = tvm.placeholder((inp_shape[0], inp_shape[1]),
                          dtype="int%d" % spec.VTA_INP_WIDTH,
                          name=SCOPE_INP)
    k = tvm.reduce_axis((0, wgt_shape[1]), name="k")
    out_dtype = "int%d" % spec.VTA_OUT_WIDTH
    out = tvm.compute((out_shape[0], out_shape[1]),
                      lambda i, j: tvm.sum(inp[i, k].astype(out_dtype) *
                                           wgt[j, k].astype(out_dtype),
                                           axis=[k]),
                      name="out")
    wgt_layout = tvm.decl_buffer(
        wgt.shape, wgt.dtype, SCOPE_WGT,
        scope=SCOPE_WGT, offset_factor=wgt_lanes, data_alignment=wgt_lanes)
    inp_layout = tvm.decl_buffer(
        inp.shape, inp.dtype, SCOPE_INP,
        scope=SCOPE_INP, offset_factor=inp_lanes, data_alignment=inp_lanes)
    out_layout = tvm.decl_buffer(
        out.shape, out.dtype, SCOPE_OUT,
        scope=SCOPE_OUT, offset_factor=out_lanes, data_alignment=out_lanes)

    def intrin_func(ins, outs):
        """Matrix-matrix multiply intrinsic function"""
        dinp, dwgt = ins
        dout = outs[0]
        def instr(index):
            """Generate matrix-matrix multiply VTA instruction"""
            irb = tvm.ir_builder.create()
            irb.scope_attr(VTA_AXIS, "coproc_scope", get_task_qid(spec.VTA_QID_COMPUTE))
            irb.scope_attr(VTA_AXIS, "coproc_uop_scope", VTA_PUSH_UOP)
            if index == 0 or index == 2:
                irb.emit(tvm.call_extern(
                    "int32", "VTAUopPush",
                    0, 0,
                    dout.access_ptr("rw", "int32"),
                    dinp.access_ptr("r", "int32"),
                    dwgt.access_ptr("r", "int32"),
                    0, 0, 0))
            else:
                irb.emit(tvm.call_extern(
                    "int32", "VTAUopPush",
                    0, 1,
                    dout.access_ptr("rw", "int32"),
                    0,
                    0,
                    0, 0, 0))
            return irb.get()
        # return a triple of normal-set, reset, update
        nop = tvm.make.Evaluate(0)
        if mock:
            return (nop, nop, nop)
        return (instr(0), instr(1), instr(2))

    return tvm.decl_tensor_intrin(out.op, intrin_func,
                                  name="GEMM",
                                  binds={inp: inp_layout,
                                         wgt: wgt_layout,
                                         out: out_layout})

GEMM = intrin_gemm()
GEVM = intrin_gevm()
