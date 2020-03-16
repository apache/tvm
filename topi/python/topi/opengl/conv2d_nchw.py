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
#pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments, too-many-branches, line-too-long
"""Schedule for conv2d_nchw with auto fusion"""
import tvm
from tvm import te
from .. import tag

def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _schedule(conv2d, data):
        if conv2d.op in s.outputs:
            Out = conv2d
        else:
            Out = outs[0].op.output(0)
            s[conv2d].opengl()
        s[Out].opengl()
        s[data].opengl()

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].opengl()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule conv2d_nchw
        elif OP.tag.startswith('conv2d_nchw'):
            conv2d = OP.output(0)
            data = OP.input_tensors[0]
            kernel = OP.input_tensors[1]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            _schedule(conv2d, data)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
