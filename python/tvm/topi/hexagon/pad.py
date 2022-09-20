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

"""Schedule for nn.pad operator"""

import tvm

import numpy as np


def schedule_pad(outs):
    """Schedule for pad op.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of injective in the format
        of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = tvm.te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)

    # Fuse axes and vectorize only if last output tensor dimension is divisible by a factor:
    factor = 128 // np.dtype(outs[0].dtype).itemsize
    last_dim = outs[0].shape[-1]
    if last_dim % factor == 0 and last_dim // factor >= 0:
        fused = s[outs[0]].fuse(*outs[0].op.axis)
        _, inner = s[outs[0]].split(fused, factor=factor)
        s[outs[0]].vectorize(inner)

    return s
