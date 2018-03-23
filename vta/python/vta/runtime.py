"""Runtime function related hooks"""
from __future__ import absolute_import as _abs

import tvm

def thread_local_command_buffer():
    """Get thread local command buffer"""
    ctx = tvm.call_extern("handle", "VTATLSCommandHandle")
    return tvm.make.Call(
        "handle", "tvm_thread_context", [ctx], tvm.expr.Call.Intrinsic, None, 0)

CB_HANDLE = thread_local_command_buffer()

VTA_AXIS = tvm.thread_axis("vta")
VTA_PUSH_UOP = tvm.make.StringImm("VTAPushGEMMOp")

SCOPE_INP = "local.inp_buffer"
SCOPE_OUT = "local.out_buffer"
SCOPE_WGT = "local.wgt_buffer"
DMA_COPY = "dma_copy"
ALU = "alu"
DEBUG_NO_SYNC = False

def get_task_qid(qid):
    """Get transformed queue index."""
    return 1 if DEBUG_NO_SYNC else qid


@tvm.register_func("tvm.intrin.rule.default.vta.coproc_sync")
def coproc_sync(op):
    return tvm.call_extern(
        "int32", "VTASynchronize", CB_HANDLE, 1<<31)

@tvm.register_func("tvm.intrin.rule.default.vta.coproc_dep_push")
def coproc_dep_push(op):
    return tvm.call_extern(
        "int32", "VTADepPush", CB_HANDLE, op.args[0], op.args[1])

@tvm.register_func("tvm.intrin.rule.default.vta.coproc_dep_pop")
def coproc_dep_pop(op):
    return tvm.call_extern(
        "int32", "VTADepPop", CB_HANDLE, op.args[0], op.args[1])
