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
"""Dense schedule for ARM CPU"""
from tvm import autotvm
from .mprofile.dsp.dense import dense_dsp_schedule, dense_dsp_compute
from .dense_gemm import dense_gemm_compute, dense_gemm_schedule


@autotvm.register_topi_compute("dense_dsp.arm_cpu")
def dense_dsp(cfg, data, weight, bias, out_dtype):
    """Compute dense with DSP instructions."""
    return dense_dsp_compute(cfg, data, weight, bias=bias, out_dtype=out_dtype)


@autotvm.register_topi_schedule("dense_dsp.arm_cpu")
def schedule_dense_dsp(cfg, outs):
    """Create schedule for dense_dsp"""
    return dense_dsp_schedule(cfg, outs)


@autotvm.register_topi_compute("dense_gemm.arm_cpu")
def dense_gemm(cfg, data, weight, bias, out_dtype, transpose_a=False, transpose_b=True):
    """Compute dense using GeMM."""
    return dense_gemm_compute(cfg, data, weight, bias, out_dtype, transpose_a, transpose_b)


@autotvm.register_topi_schedule("dense_gemm.arm_cpu")
def schedule_dense_gemm(cfg, outs):
    """Create schedule for dense using GeMM."""
    return dense_gemm_schedule(cfg, outs)
