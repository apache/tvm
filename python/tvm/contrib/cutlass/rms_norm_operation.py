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
"""Generator for CUTLASS rms norm kernels."""
from .library import substitute_template


def instantiate_rms_norm_template(attrs):
    """
    Return CUTLASS host code for rms norm based on
    a template and the provided attribute map.
    """
    template = """
    using data_type = ${data_type};
    using namespace cutlass::layout;

    int M = ${M};
    int N = ${N};
    cutlass::MatrixCoord size(M, N);
    auto layout_2D = RowMajor::packed(size);
    auto layout_channels = RowMajor::packed({1, N});

    cutlass::TensorRef<data_type, RowMajor> _input((data_type*)${input}->data, layout_2D);
    cutlass::TensorRef<data_type, RowMajor> _weight((data_type*)${weight}->data, layout_channels);
    cutlass::TensorRef<data_type, RowMajor> _output((data_type*)out0->data, layout_2D);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    cutlass::rmsnorm(size, _output, _input, _weight, stream, ${rms_eps});
    """
    return substitute_template(template, attrs)
