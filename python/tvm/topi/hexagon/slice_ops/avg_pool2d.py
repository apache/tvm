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

from tvm.ir.module import IRModule
from tvm import te
from tvm import tir
from tvm.script import tir as T
from ..utils import apply_transform, get_layout_transform_fn


# The slice op implementation for avg_pool2d makes serveral assumptions:
# 1) Both input and output are a multiple of croutons, and the input is already
#    padded for a given output shape as per any crouton and non-crouton related
#    padding.
# 2) The current implementation assumes 'count_include_pad' to be 'True'. It can
#    modified to support 'False' but the element count for the pooling window must
#    be pre-computed and provided as an input to reduce the run-time overhead.
# 3) 'padding' is also ignored. It must be handled outside of the sliced op.
# 4) Please note that this implementation will not work if the output was padded
#    for the croutons. Since we loop over the logical output shape, this can result
#    into out-of-bound access for the input.

def avg_pool2d_compute(A, out_shape, kernel, stride, dilation):
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")
    ob, oh, ow, oc = out_shape
    sh, sw = stride
    dh, dw = dilation
    Area = float(1) / (kh * kw)

    Sum = te.compute(
        out_shape,
        lambda b, h, w, c: te.sum(
            A[b, h * sh + dh * rh, w * sw + dw * rw, c].astype("float32"), axis=[rh, rw]
        ),
        name="sum",
    )
    Avg = te.compute(
        out_shape, lambda b, h, w, c: (Sum[b, h, w, c] * Area).astype(A.dtype), name="avg"
    )
    return Avg


# Schedule for input and output layout nhwc-8h2w32c2w
def STIR_schedule_nhwc_8h2w32c2w(outs, ins, output_layout: str, input_layout: str):
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("sum")
    Avg = s.get_block("avg")

    apply_transform(s, Sum, 0, "read", input_layout)
    apply_transform(s, Avg, 0, "write", output_layout)

    # Schedule 'Sum'
    bn, bh, bw, bc, rx, ry = s.get_loops(Sum)
    bho, bhi = s.split(bh, [None, 8])
    bwo, bwi = s.split(bw, [None, 4])
    bwio, bwii = s.split(bwi, [None, 2])  # Doesn't seem to be doing anything
    bco, bci = s.split(bc, [None, 32])
    s.reorder(bn, bho, bwo, bco, bhi, bwio, rx, ry, bci, bwii)  # --- DOESN'T do anything
    bci_wii = s.fuse(bci, bwii)  # --- DOESN'T do anything
    # s.vectorize(bci_wii) # --- DOESN'T WORK -- errors out

    # Schedule 'Avg'
    n, h, w, c = s.get_loops(Avg)
    ho, hi = s.split(h, [None, 8])
    wo, wi = s.split(w, [None, 4])
    wio, wii = s.split(wi, [None, 2])
    co, ci = s.split(c, [None, 32])
    s.reorder(n, ho, wo, co, hi, wio, ci, wii)
    ci_wii = s.fuse(ci, wii)
    s.vectorize(ci_wii)

    s.compute_at(Sum, hi)
    return s


# Schedule for output layout: n11c-1024c, input layout: nhwc-8h2w32c2w
def STIR_schedule_n11c_1024c(outs, ins, output_layout: str, input_layout: str):
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("sum")
    Avg = s.get_block("avg")

    apply_transform(s, Sum, 0, "read", input_layout)
    apply_transform(s, Avg, 0, "write", output_layout)

    bn, bh, bw, bc, rx, ry = s.get_loops(Sum)
    bco, bci = s.split(bc, [None, 1024])
    bcio, bcii = s.split(bci, [None, 64])
    s.reorder(bn, bh, bw, bco, bcio, rx, ry, bcii)  # --- DOESN'T do anything
    # s.vectorize(bcii) # --- DOESN'T WORK -- errors out

    n, h, w, c = s.get_loops(Avg)
    co, ci = s.split(c, [None, 1024])
    cio, cii = s.split(ci, [None, 64])
    s.vectorize(cii)

    s.compute_at(Sum, cio)
    return s


# TIR based schedule
def avg_pool2d_STIR_schedule(outs, ins, output_layout: str, input_layout: str):
    output_layout += "-1d"
    input_layout += "-1d"
    if output_layout == "nhwc-8h2w32c2w-1d":
        return STIR_schedule_nhwc_8h2w32c2w(outs, ins, output_layout, input_layout)
    if output_layout == "n11c-1024c-1d":
        return STIR_schedule_n11c_1024c(outs, ins, output_layout, input_layout)
    else:
        raise RuntimeError(f"Unexpected layout '{output_layout}'")


# Schedule for input and output layout nhwc-8h2w32c2w
def schedule_nhwc_8h2w32c2w(outs, ins, output_layout: str, input_layout: str):
    A = ins
    M = outs
    s = te.create_schedule([M.op])
    B = s[M].op.input_tensors[0]

    # Apply layout transformation
    input_layout = get_layout_transform_fn(input_layout)
    output_layout = get_layout_transform_fn(output_layout)
    s[A].transform_layout(input_layout)
    M_axis = s[M].transform_layout(output_layout)

    # Schedule 'M'
    m_inner = s[M].fuse(M_axis[7], M_axis[6])
    s[M].vectorize(m_inner)

    # Schedule 'B'
    bn, bh, bw, bc = s[B].op.axis
    rx, ry = s[B].op.reduce_axis
    bwo, bwi = s[B].split(bw, factor=4)
    bwio, bwii = s[B].split(bwi, factor=2)
    bco, bci = s[B].split(bc, factor=32)
    s[B].reorder(bn, bco, bh, bwo, bwio, ry, rx, bci, bwii)
    b_inner = s[B].fuse(bci, bwii)
    # s[B].vectorize(b_inner) # Doesn't work

    s[B].compute_at(s[M], M_axis[5])
    return s


# Schedule for output layout: n11c-1024c, input layout: nhwc-8h2w32c2w
def schedule_n11c_1024c(outs, ins, output_layout: str, input_layout: str):
    A = ins
    M = outs
    s = te.create_schedule([M.op])
    B = s[M].op.input_tensors[0]

    # Apply layout transformation
    input_layout = get_layout_transform_fn(input_layout)
    output_layout = get_layout_transform_fn(output_layout)
    s[A].transform_layout(input_layout)
    M_axis = s[M].transform_layout(output_layout)

    # Schedule 'M'
    mco, mci = s[M].split(M_axis[4], factor=64)
    s[M].vectorize(mci)

    # Schedule 'B'
    bn, bh, bw, bc = s[B].op.axis
    rx, ry = s[B].op.reduce_axis
    bco, bci = s[B].split(bc, factor=64)
    s[B].reorder(bco, rx, ry, bci)
    # s[B].vectorize(bci) # Doesn't work

    s[B].compute_at(s[M], mco)
    return s


# te based schedule
def avg_pool2d_schedule(outs, ins, output_layout: str, input_layout: str):
    output_layout += "-2d"
    input_layout += "-2d"
    if output_layout == "nhwc-8h2w32c2w-2d":
        return schedule_nhwc_8h2w32c2w(outs, ins, output_layout, input_layout)
    if output_layout == "n11c-1024c-2d":
        return schedule_n11c_1024c(outs, ins, output_layout, input_layout)
    else:
        raise RuntimeError(f"Unexpected layout '{output_layout}'")
