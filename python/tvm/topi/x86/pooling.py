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
# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
from tvm import te
from .. import tag


def _parallel_sch(sch, oshape, do_vectorize=False):
    def vectorize(fused_axis, num_parallel_axis, vectorize_limit=64):
        """Internal vectorization utility function."""
        reorder_axis = [fused_axis]
        for i in range(num_parallel_axis, len(sch.op.axis) - 1):
            reorder_axis.append(sch.op.axis[i])
        k = sch.op.reduce_axis
        fuse_k = sch.fuse(*k)
        c = sch.op.axis[len(sch.op.axis) - 1]
        reorder_axis += [fuse_k, c]
        sch.reorder(*reorder_axis)
        inner_length = oshape[len(oshape) - 1].value
        if inner_length <= vectorize_limit:
            sch.vectorize(c)
        else:
            split_factor = 1
            for i in range(vectorize_limit, 1, -1):
                if inner_length % i == 0:
                    split_factor = i
                    break
            if split_factor > 1:
                _, c_i = sch.split(c, split_factor)
                sch.vectorize(c_i)

    if len(sch.op.axis) >= 5:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1], sch.op.axis[2])
        if do_vectorize:
            vectorize(fused, 3)

    elif len(sch.op.axis) >= 3:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1])
        if do_vectorize:
            vectorize(fused, 2)
    else:
        sch.parallel(sch.op.axis[0])
        return
    sch.parallel(fused)


def schedule_pool(outs, layout):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    layout: str
        Data layout.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, te.tensor.ComputeOp):
            s[PaddedInput].compute_inline()
        do_vectorize = layout[-1] not in "DHWdhw"
        _parallel_sch(s[Pool], outs[0].shape, do_vectorize)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith("pool"):
            # Average pool accumulation and division happens in different for loops (#3607).
            # To ensure good parallel support, apply multi-threading on the second loop.
            if OP != outs[0].op:
                output = outs[0]
                output_fused = s[output].fuse(output.op.axis[0], output.op.axis[1])
                s[output].parallel(output_fused)

            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


def schedule_adaptive_pool(outs):
    """Schedule for adaptive pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of adaptive pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith("adaptive_pool"):
            if OP != outs[0].op:
                output = outs[0]
                output_fused = s[output].fuse(output.op.axis[0], output.op.axis[1])
                s[output].parallel(output_fused)

            Pool = OP.output(0)
            _parallel_sch(s[Pool], outs[0].shape)
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
