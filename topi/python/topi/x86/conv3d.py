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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin, no-else-return
"""Conv3D operators"""
import tvm
from .. import generic, tag
from ..util import traverse_inline

@generic.schedule_conv3d_ndhwc.register("cpu")
def schedule_conv3d_ndhwc(outs):
    """TOPI schedule callback for conv3d

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv3d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv3d.
    """
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op

    def _traverse(op):
        """Traverse operators from computation graph"""
        if op in s.outputs and tag.is_broadcast(op.tag) and len(op.axis) == 5:
            # schedule bias + bn + relu
            n, d, h, w, c = op.axis
            fused = s[op].fuse(n, d, h, w)
            s[op].parallel(fused)
            s[op].vectorize(c)

        if 'conv3d_ndhwc' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            # dilation stage
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            # padding stage
            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                # fuse pad h and w
                data_pad = data
                data = data_pad.op.input_tensors[0]
                _, _, h_pad, w_pad, _ = data_pad.op.axis
                pad_fused = s[data_pad].fuse(h_pad, w_pad)
                s[data_pad].parallel(pad_fused)

            # compute conv
            C = conv
            n, d, h, w, c = s[C].op.axis
            s[C].vectorize(c)
            if op != output_op: # fuse bias + bn + activation
                _, _, _, _, c_out = output_op.axis
                s[C].compute_at(s[output_op], c_out)
            else:
                # fuse batch, depth, height axes
                fused = s[C].fuse(n, d, h)
                s[C].parallel(fused)

    traverse_inline(s, output_op, _traverse)
    return s
