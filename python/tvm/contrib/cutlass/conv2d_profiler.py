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
# pylint: disable=import-outside-toplevel, invalid-name
"""Instantiate a C++ source for profiling CUTLASS kernels."""
from .gemm_profiler import GemmProfilerEmitter


class Conv2dProfilerEmitter:
    def __init__(self):
        self.gemm_profiler_emitter = GemmProfilerEmitter()

    def emit(self, op_name, op_def, dtype_a, dtype_b, dtype_c, ld):
        return self.gemm_profiler_emitter(op_name, op_def, dtype_a, dtype_b, dtype_c, ld)
