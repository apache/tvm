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

from .library import DataTypeTag


class Conv2dProfilerEmitter(object):
    """Emit a C++ source for profiling CUTLASS kernels."""

    def __init__(self):
        from jinja2 import Template

        self.reduction = """
      ReductionDevice reduction_op;
      static cutlass::conv::Operator const kConvolutionalOperator = ImplicitGemm::kConvolutionalOperator;
      typename ReductionDevice::Arguments reduction_args(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
        problem_size.split_k_slices,
        cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
        {
      	 reinterpret_cast<ImplicitGemm::ElementC*> (workspace.get()),
      	   ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
      	   },
        {
      	 tensor_d.device_data(),
      	   ReductionStrideIndex(tensor_d.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
      	   },
        {
      	 tensor_c.device_data(),
      	   ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
      	   },
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}
        );

      reduction_op.initialize(reduction_args, nullptr);
      reduction_op();
"""

        self.template = Template(
            """
#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

{{OperatorDef}}
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<{{OperatorName}}>;

struct Options {
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());
    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
    cmd.get_cmd_line_argument("pad_h", pad_h);
    cmd.get_cmd_line_argument("pad_w", pad_w);
    cmd.get_cmd_line_argument("stride_h", stride_h);
    cmd.get_cmd_line_argument("stride_w", stride_w);
    cmd.get_cmd_line_argument("dilation_h", dilation_h);
    cmd.get_cmd_line_argument("dilation_w", dilation_w);
    filter_size.c() = input_size.c();
    padding = {pad_h, pad_h, pad_w, pad_w};
    conv_stride = {stride_h, stride_w};
    dilation = {dilation_h, dilation_w};
  }

  cutlass::Tensor4DCoord output_size() const {
    auto dilated_h = (filter_size.h() - 1) * dilation.row() + 1;
    auto dilated_w = (filter_size.w() - 1) * dilation.column() + 1;
    auto h = (input_size.h() + padding.n() + padding.h() - dilated_h) / conv_stride.row() + 1;
    auto w = (input_size.w() + padding.w() + padding.c() - dilated_w) / conv_stride.column() + 1;
    return cutlass::Tensor4DCoord(input_size.n(), h, w, filter_size.n());
  }
};

double profile_convolution(Options const &options) {
  using ElementOutput = {{ElementOutput}};
  using ElementInputA = typename ImplicitGemm::ElementA;
  using ElementInputB = typename ImplicitGemm::ElementB;

  int split_k_slices = {{SplitK}};
  cutlass::conv::Conv2dProblemSize problem_size(
						options.input_size,
						options.filter_size,
						options.padding,
						options.conv_stride,
						options.dilation,
						options.output_size(),
						cutlass::conv::Mode::kCrossCorrelation,
						split_k_slices
						);

  auto conv_kind = ImplicitGemm::kConvolutionalOperator;
  auto a_extent = implicit_gemm_tensor_a_extent(conv_kind, problem_size);
  auto b_extent = implicit_gemm_tensor_b_extent(conv_kind, problem_size);
  auto c_extent = implicit_gemm_tensor_c_extent(conv_kind, problem_size);

  using LayoutC = typename ImplicitGemm::LayoutC;
  cutlass::HostTensor<ElementInputA, typename ImplicitGemm::LayoutA> tensor_a(a_extent);
  cutlass::HostTensor<ElementInputB, typename ImplicitGemm::LayoutB> tensor_b(b_extent);
  cutlass::HostTensor<ElementOutput, typename ImplicitGemm::LayoutC> tensor_c(c_extent);
  cutlass::HostTensor<ElementOutput, LayoutC> tensor_d(c_extent);
  cutlass::HostTensor<ImplicitGemm::ElementC, LayoutC> tensor_c_gemm(c_extent);

  using ElementComputeEpilogue = typename ImplicitGemm::ElementCompute;

  cutlass::conv::SplitKMode const split_k_mode = split_k_slices > 1 ?
      cutlass::conv::SplitKMode::kParallel : cutlass::conv::SplitKMode::kSerial;

  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c_gemm.device_ref(),
    tensor_c_gemm.device_ref(),
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},
    split_k_mode,
  };

  ImplicitGemm implicit_gemm_op;
  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  auto status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  status = implicit_gemm_op();
  CUTLASS_CHECK(status);

  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0]);

  for (int iteration = 0; iteration < 100; ++iteration) {
    auto status = implicit_gemm_op();
    CUTLASS_CHECK(status);
    {{Reduction}}
  }

  cudaEventRecord(events[1]);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);

  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  return double(runtime_ms) / 100.0;
}

int main(int argc, char const **args) {
  Options options;
  options.parse(argc, args);
  std::cout << profile_convolution(options) << std::endl;
  return 0;
}
"""
        )

    def emit(self, op_def, op_name, element_output, split_k_slices=1):
        src = self.template.render(
            OperatorDef=op_def,
            OperatorName=op_name,
            ElementOutput=DataTypeTag[element_output],
            SplitK=split_k_slices,
            Reduction=self.reduction if split_k_slices > 1 else "",
        )
        return src
