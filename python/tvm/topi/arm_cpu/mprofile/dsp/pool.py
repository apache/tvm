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
import logging

import tvm

from tvm import te
from tvm.topi.utils import traverse_inline

from .micro_kernel.max_pool import (
    intrin_max,
    max_impl,
)

from .micro_kernel.avg_pool import (
    intrin_sum,
    sum_impl,
)

logger = logging.getLogger("topi")


def schedule_maxpool_1d_nwc(s, op):
    """Schedule function for v7e-m DSP instructions of maxpool 1d NWC layout."""
    output = op.output(0)
    data_vec = op.input_tensors[0]

    channels = data_vec.shape[-1]
    if isinstance(channels, tvm.tir.IntImm):
        channels = channels.value

    n, w, c = s[op].op.axis
    (k,) = s[op].op.reduce_axis

    s[op].reorder(n, w, k, c)
    max_val, uniq_id = intrin_max((1, 1, channels), data_vec.dtype, output.dtype)
    s[op].tensorize(c, max_val)
    s[output].pragma(n, "import_c", max_impl(uniq_id))


def schedule_maxpool_2d_nhwc(s, op):
    """Schedule function for v7e-m DSP instructions of maxpool 2d NHWC layout."""
    output = op.output(0)
    data_vec = op.input_tensors[0]

    channels = data_vec.shape[-1]
    if isinstance(channels, tvm.tir.IntImm):
        channels = channels.value

    n, h, w, c = s[op].op.axis
    ko, ki = s[op].op.reduce_axis

    s[op].reorder(n, h, w, ko, ki, c)
    max_val, uniq_id = intrin_max((1, 1, 1, channels), data_vec.dtype, output.dtype)
    s[op].tensorize(c, max_val)
    s[output].pragma(n, "import_c", max_impl(uniq_id))


def schedule_avgpool_1d_ncw(s, op):
    """Schedule function for v7e-m DSP instructions of avgpool 1d NCW layout."""
    output = op.output(0)
    data_vec = op.input_tensors[0]

    n, _, _ = s[op].op.axis
    (k,) = s[op].op.reduce_axis
    pool_w = k.dom.extent.value

    summary, uniq_id = intrin_sum((1, 1, pool_w), data_vec.dtype, output.dtype, reset=True)
    s[op].tensorize(k, summary)
    s[output].pragma(n, "import_c", sum_impl(pool_w, uniq_id))


def schedule_avgpool_2d_nchw(s, op):
    """Schedule function for v7e-m DSP instructions of avgpool 2d NCHW layout."""
    output = op.output(0)
    data_vec = op.input_tensors[0]

    n, _, _, _ = s[op].op.axis
    _, ki = s[op].op.reduce_axis
    pool_w = ki.dom.extent.value

    summary, uniq_id = intrin_sum((1, 1, 1, pool_w), data_vec.dtype, output.dtype)
    s[op].tensorize(ki, summary)
    s[output].pragma(n, "import_c", sum_impl(pool_w, uniq_id))


def pool_dsp_schedule(outs, layout):
    """Schedule function for v7e-m DSP instructions of pooling."""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        in_dtype = op.input_tensors[0].dtype
        if "pool_max" in op.tag:
            if in_dtype != "int8":
                logger.warning("Does not have micro-kernel for %s maxpool.", in_dtype)
            elif layout == "NWC":
                schedule_maxpool_1d_nwc(s, op)
            elif layout == "NHWC":
                schedule_maxpool_2d_nhwc(s, op)
        elif "pool_sum" in op.tag:
            if in_dtype != "int16":
                logger.warning("Does not have micro-kernel for %s avgpool.", in_dtype)
            elif layout == "NCW":
                schedule_avgpool_1d_ncw(s, op)
            elif layout == "NCHW":
                schedule_avgpool_2d_nchw(s, op)

    traverse_inline(s, outs[-1].op, _callback)
    return s
