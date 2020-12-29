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
from tvm import te


def schedule_binarize_pack(outs):
    """Schedule for binarize_pack.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of binarize_pack
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for binarize_pack.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(Out):
        s[Out].parallel(Out.op.axis[0])

    def traverse(OP):
        # schedule binarize_pack
        if OP.tag == "binarize_pack":
            Out = OP.output(0)
            _schedule(Out)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
