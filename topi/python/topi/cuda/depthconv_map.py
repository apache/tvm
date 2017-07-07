# pylint: disable=invalid-name, wildcard-import, missing-docstring
"""Depthwise convolution operator.

Auto fusion with one-to-one-mapping operators, e.g. scale-shift and relu.
"""

import os
import tvm
from tvm.contrib import nvcc_compiler
from topi.util import *

TASK = "depthconv_map"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc_compiler.compile_source(code, target="ptx", options=["-arch=sm_52"])
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

def schedule_depthconv_map(op):
    s = tvm.create_schedule(op)
    def schedule_depthconv(PaddedInput, Filter, DepthConv):
        out_shape = get_const_tuple(DepthConv.shape)
        out_height = out_shape[2]
        out_width = out_shape[3]
        channel_multiplier = get_const_tuple(Filter.shape)[1]
        s[PaddedInput].compute_inline()
        IS = s.cache_read(PaddedInput, "shared", [DepthConv])
        FS = s.cache_read(Filter, "shared", [DepthConv])
        IL = s.cache_read(IS, "local", [DepthConv])
        FL = s.cache_read(FS, "local", [DepthConv])
        if is_output(DepthConv.op, s):
            Output = DepthConv
            CL = s.cache_write(DepthConv, "local")
        else:
            Output = op.output(0)
            s[DepthConv].set_scope("local")
        # schedule parameters
        num_thread = 8
        num_vthread_x = 1
        num_vthread_y = 1
        blocking_h = out_height
        blocking_w = out_width
        if out_height % 48 == 0:
            blocking_h = 48
        elif out_height % 32 == 0:
            blocking_h = 32
        if out_width % 48 == 0:
            blocking_w = 48
            num_vthread_y = 3
        elif out_width % 32 == 0:
            blocking_w = 32
        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
        thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")
        # split and bind
        bx, bxi = s[Output].split(Output.op.axis[1], factor=channel_multiplier)
        s[Output].reorder(Output.op.axis[2], Output.op.axis[3], bxi)
        bx = s[Output].fuse(bx, Output.op.axis[0])
        s[Output].bind(bx, block_x)
        by1, y1i = s[Output].split(Output.op.axis[2], factor=blocking_h)
        tvx, vxi = s[Output].split(y1i, nparts=num_vthread_x)
        tx, xi = s[Output].split(vxi, nparts=num_thread)
        by2, y2i = s[Output].split(Output.op.axis[3], factor=blocking_w)
        tvy, vyi = s[Output].split(y2i, nparts=num_vthread_y)
        ty, yi = s[Output].split(vyi, nparts=num_thread)
        s[Output].reorder(by1, by2, tvx, tvy, tx, ty, xi, yi)
        by = s[Output].fuse(by2, by1)
        s[Output].bind(tvx, thread_vx)
        s[Output].bind(tvy, thread_vy)
        s[Output].bind(tx, thread_x)
        s[Output].bind(ty, thread_y)
        s[Output].bind(by, block_y)
        # local memory load
        s[IL].compute_at(s[Output], ty)
        s[FL].compute_at(s[Output], ty)
        if is_output(DepthConv.op, s):
            s[CL].compute_at(s[Output], ty)
        else:
            s[DepthConv].compute_at(s[Output], ty)
        # input's shared memory load
        s[IS].compute_at(s[Output], by)
        tx, xi = s[IS].split(IS.op.axis[2], nparts=num_thread)
        ty, yi = s[IS].split(IS.op.axis[3], nparts=num_thread)
        s[IS].bind(tx, thread_x)
        s[IS].bind(ty, thread_y)
        # filter's shared memory load
        s[FS].compute_at(s[Output], by)
        s[FS].reorder(FS.op.axis[2], FS.op.axis[3], FS.op.axis[1])
        tx, xi = s[FS].split(FS.op.axis[2], nparts=num_thread)
        ty, yi = s[FS].split(FS.op.axis[3], nparts=num_thread)
        s[FS].bind(tx, thread_x)
        s[FS].bind(ty, thread_y)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if OP.tag == 'ewise' or OP.tag == 'scale_shift':
            if not is_output(OP, s):
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if str(tensor.op.input_tensors) != str([]):
                    traverse(tensor.op)
        # schedule depthconv
        if OP.tag == 'depthconv':
            PaddedInput = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            DepthConv = OP.output(0)
            schedule_depthconv(PaddedInput, Filter, DepthConv)

    traverse(op)
    return s
