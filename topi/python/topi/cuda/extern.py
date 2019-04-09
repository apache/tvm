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
# pylint: disable=invalid-name, unused-variable,
"""Schedule for cudnn and miopen extern op"""
import tvm
from .. import generic
from .injective import _schedule_injective


@generic.schedule_extern.register(["cuda", "gpu"])
def schedule_extern(outs):
    """Schedule for an extern op followed by injective operations.
       For example, cudnn kernel + bias add + relu.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of extern plus injective ops in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    tvm.schedule.AutoInlineInjective(s)
    for out in outs:
        if isinstance(out.op, tvm.tensor.ExternOp):
            continue
        _schedule_injective(out.op, s)
    return s
