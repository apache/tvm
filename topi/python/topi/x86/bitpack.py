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
"""Schedule for binarization and bit-packing."""
from __future__ import absolute_import as _abs
import tvm
from .. import generic


@generic.schedule_bitpack.register(["cpu"])
def schedule_bitpack(outs):
    """Schedule for bitpack.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of bitpack 
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for bitpack.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Out):
        s[Out].parallel(Out.op.axis[0])

    def traverse(OP):
        # schedule bitpack
        if OP.tag == 'bitpack':
            Out = OP.output(0)
            _schedule(Out)

    traverse(outs[0].op)
    return s
