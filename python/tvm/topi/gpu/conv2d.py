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
# pylint: disable=invalid-name, unused-argument
"""Schedule for conv2d operator"""
from tvm import te, autotvm

from .. import nn
from ..utils import traverse_inline
from .conv2d_nhwc import schedule_conv2d_nhwc_direct


@autotvm.register_topi_compute("conv2d_nhwc.gpu")
def conv2d_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d with NHWC layout"""
    return nn.conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc.gpu")
def schedule_conv2d_nhwc(cfg, outs):
    """Create the schedule for conv2d_nhwc"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv2d_nhwc":
            schedule_conv2d_nhwc_direct(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
