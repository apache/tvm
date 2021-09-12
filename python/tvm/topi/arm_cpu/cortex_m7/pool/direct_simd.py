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
# pylint: disable=invalid-name, no-value-for-parameter
"""Direct implementation of pool."""
import tvm

from tvm import te
from tvm.topi.utils import simplify, traverse_inline

from ..micro_kernel.max_pool import (
    intrin_max,
    max_pool_impl,
)


def pool_direct_simd_nhwc_schedule(outs):
    """Schedule function for Cortex-M7 SIMD implementation of max_pool2d."""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "pool_max" not in op.tag:
            return

        # extract tensors
        output = op.output(0)
        data_vec = op.input_tensors[0]

        in_channels = data_vec.shape[-1]
        if isinstance(in_channels, tvm.tir.IntImm):
            in_channels = in_channels.value

        n, h, w, c = s[op].op.axis
        ko, ki = s[op].op.reduce_axis

        if "pool_max" in op.tag:
            s[op].reorder(n, h, w, ko, ki, c)
            max, uniq_id = intrin_max(in_channels, data_vec.dtype, output.dtype)
            s[op].tensorize(c, max)
            s[output].pragma(n, "import_c", max_pool_impl(uniq_id))

    traverse_inline(s, outs[-1].op, _callback)
    return s
