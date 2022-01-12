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
# pylint: disable=invalid-name, unused-variable, no-else-return, unused-argument, import-outside-toplevel
"""Conv1D schedule for ARM CPU"""
from __future__ import absolute_import as _abs

from tvm import autotvm

from .mprofile.dsp.conv1d import (
    conv1d_nwc_dsp_compute,
    conv1d_nwc_dsp_schedule,
)


@autotvm.register_topi_compute("conv1d_nwc_dsp.arm_cpu")
def conv1d_nwc_dsp(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv1d with v7e-m DSP instructions."""
    return conv1d_nwc_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv1d_nwc_dsp.arm_cpu")
def schedule_conv1d_nwc_dsp(cfg, outs):
    return conv1d_nwc_dsp_schedule(cfg, outs)
