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
# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
from tvm import te
from tvm.contrib import cudnn
from .. import generic
from .injective import schedule_injective_from_existing


def schedule_softmax(outs):
    """Schedule for softmax op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    softmax = outs[0]

    op_tag = softmax.op.tag
    if op_tag == 'softmax_output':
        expsum = softmax.op.input_tensors[1]
        exp = softmax.op.input_tensors[0]
        max_elem = s[exp].op.input_tensors[1]
    elif op_tag == 'log_softmax_output':
        exp = None
        max_elem = softmax.op.input_tensors[1]
        expsum = softmax.op.input_tensors[2]
    else:
        raise ValueError('Tag is expected to be softmax_output or log_softmax_output. \
                         Got {0}'.format(op_tag))

    if len(softmax.shape) > 2:
        ops = [max_elem.op, expsum.op, softmax.op]
        if exp is not None:
            ops.append(exp.op)

        for op in ops:
            s = schedule_injective_from_existing(s, op.output(0))
    else:
        num_thread = 64
        block_x = te.thread_axis("blockIdx.x")
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")

        if exp is not None:
            s[exp].bind(exp.op.axis[0], block_x)

        s[max_elem].bind(max_elem.op.axis[0], block_x)
        k = expsum.op.reduce_axis[0]
        ko, ki = s[expsum].split(k, factor=num_thread)
        EF = s.rfactor(expsum, ki)
        s[expsum].bind(s[expsum].op.axis[0], block_x)
        s[expsum].bind(s[expsum].op.reduce_axis[0], thread_x)
        s[EF].compute_at(s[expsum], s[expsum].op.reduce_axis[0])
        s[expsum].set_store_predicate(thread_x.var.equal(0))
        tx, xi = s[softmax].split(softmax.op.axis[1], nparts=num_thread)
        s[softmax].bind(softmax.op.axis[0], block_x)
        s[softmax].bind(tx, thread_x)

    return s


def softmax_cudnn(x, axis=-1):
    """Perform softmax on the data using cudnn"""
    return cudnn.softmax(x, axis)


def schedule_softmax_cudnn(outs):
    """Schedule for softmax cudnn op"""
    return generic.schedule_extern(outs)
