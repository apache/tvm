/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/op/nn/convolution.h
 * \brief Properties def of convlution operator for sharing.
 */
#ifndef TVM_RELAY_OP_NN_CONVOLUTION_H_
#define TVM_RELAY_OP_NN_CONVOLUTION_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>

#include <string>
#include <utility>
#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relay {

bool Conv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter);

bool Conv2DTransposeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter);

template <typename AttrType>
bool Conv2DWinogradRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const AttrType* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = tir::BijectiveLayout(in_layout, kNCHW);
  ICHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = tir::BijectiveLayout(kernel_layout, kOIHW);
  ICHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIHW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = tir::BijectiveLayout(out_layout, kNCHW);
  ICHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;

  ICHECK(param->kernel_size.defined() && param->channels.defined())
      << "The kernel size and channels of a Conv must be set or inferred by previous pass";

  ICHECK_EQ(param->kernel_size.size(), 2);
  ICHECK_EQ(param->dilation.size(), 2);

  channels = param->channels;
  dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
  dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];

  // NOTE: Do not check weight shape here!
  // Different backend requires different layout to compute
  // the batch gemm stage in winograd efficiently, but we want to
  // make this op work for all backends.
  // So we accept all weight shapes, and assume the TOPI developers
  // can handle this correctly in alter_op_layout.

  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  if (!dshape_nchw[2].as<tir::AnyNode>()) {
    oshape.Set(2, (dshape_nchw[2] + pad_h - dilated_ksize_y) / param->strides[0] + 1);
  } else {
    oshape.Set(2, dshape_nchw[2]);
  }
  if (!dshape_nchw[3].as<tir::AnyNode>()) {
    oshape.Set(3, (dshape_nchw[3] + pad_w - dilated_ksize_x) / param->strides[1] + 1);
  } else {
    oshape.Set(3, dshape_nchw[3]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  oshape = trans_out_layout.BackwardShape(oshape);
  // assign output type
  reporter->Assign(types[2], TensorType(oshape, out_dtype));
  return true;
}

template <typename T>
InferCorrectLayoutOutput ConvInferCorrectLayout(const Attrs& attrs,
                                                const Array<Layout>& new_in_layouts,
                                                const Array<Layout>& old_in_layouts,
                                                const Array<tvm::relay::Type>& old_in_types) {
  const T* params = attrs.as<T>();
  // We always make other operators to fit the layouts of convolution layers
  // So this inference ignores all inputs
  return InferCorrectLayoutOutput(
      {params->data_layout, params->kernel_layout},
      {params->out_layout == "" ? params->data_layout : params->out_layout}, attrs);
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_CONVOLUTION_H_
