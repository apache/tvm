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

# pylint: disable=invalid-name

"""
Clip the elements in `A` between `A_min` and `A_max`.
"""

from tvm import te, tir, topi
from ..utils import get_layout_transform_fn
from tvm.topi.utils import traverse_inline

def clip_compute_flat(A, A_min, A_max):
    """
    This compute expects the input tensor to be in flat layout.
    It pads the input to ensure it is a multiple of croutons.
    After computing clip, it removes the padding.
    """

    block_H = 8
    block_W = 8
    block_C = 32

    N, H, W, C = A.shape
    pad_h = (block_H - (H % block_H)) % block_H
    pad_w = (block_W - (W % block_W)) % block_W

    # pad
    A_pad = topi.nn.pad(
        A, [0, 0, 0, 0], [0, pad_h, pad_w, 0], pad_value=0, name="pad_input"
    )

    # perform clip
    M_padded = topi.clip(A_pad, A_min, A_max)

    # remove padding
    M = topi.nn.pad(
        M_padded, [0, 0, 0, 0], [0, -pad_h, -pad_w, 0], pad_value=0, name="remove_padding"
    )
    print("In clip_compute_flat")

    return M


def clip_schedule_flat(outs):
    """
    This schedule expects the input tensor to be in flat layout.
    It assumes the layout is in NHWC format and the crouton layout is nhwc-8h2w32c2w-2d.
    Only one layout is needed because this is an elementwise operation that does not change the shape of the input.
    The schedule inserts cache reads and writes to utilize VTCM.
    """

    import pdb; pdb.set_trace()
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    print("in clip_schedule_flat")

    def _callback(op):
        if "injective" in op.tag:
            M = op.output(0)
            A = op.input_tensors[0]

    traverse_inline(s, output_op, _callback)
    return s


def clip_compute_crouton(A, A_min, A_max):
    """
    Use topi clip implementation
    Expects input to already be a multiple of croutons
    """
    return topi.clip(A, A_min, A_max)


def clip_te_schedule_crouton(outs):
    """
    This schedule expects the input tensor to be in flat layout.
    It assumes the layout is in NHWC format and the crouton layout is nhwc-8h2w32c2w-2d.
    Only one layout is needed because this is an elementwise operation that does not change the shape of the input.
    The schedule inserts cache reads and writes to utilize VTCM.
    """

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    print("in clip_te_schedule_crouton")

    def _callback(op):
        if "elemwise" in op.tag:
            O = op.output(0)
            I = op.input_tensors[0]
            input_layout = "nhwc-8h2w32c2w-2d"
            output_layout = "nhwc-8h2w32c2w-2d"
            input_layout = get_layout_transform_fn(input_layout)
            output_layout = get_layout_transform_fn(output_layout)
            s[I].transform_layout(input_layout)
            s[O].transform_layout(output_layout)

    traverse_inline(s, output_op, _callback)
    return s

def clip_schedule_crouton(outs, ins, output_layout: str, input_layout: str):
    """
    Hexagon clip schedule
    Expects input to already be a multiple of croutons
    """
    A = ins
    M = outs

    func = te.create_prim_func([A, M])

    s = tir.Schedule(func)

    block = s.get_block("compute")

    input_transformed_layout = get_layout_transform_fn(input_layout)
    s.transform_layout(block, buffer=("read", 0), index_map=input_transformed_layout)

    output_transformed_layout = get_layout_transform_fn(output_layout)
    s.transform_layout(block, buffer=("write", 0), index_map=output_transformed_layout)

    n, h, w, c = s.get_loops(block)

    ho, hi = s.split(h, [None, 8])
    wo, wi = s.split(w, [None, 4])
    co, ci = s.split(c, [None, 32])
    wio, wii = s.split(wi, [None, 2])

    s.reorder(n, ho, wo, co, hi, wio, ci, wii)

    fused = s.fuse(ci, wii)
    s.vectorize(fused)

    return s
