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

#include <tvm/tir/ir_pass.h>
#include <string>
#include <utility>

#include "../op_common.h"

namespace tvm {
namespace relay {

template <typename AttrType>
bool Conv1DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCW("NCW");
  static const Layout kOIW("OIW");

  const AttrType* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCW);
  CHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = BijectiveLayoutNode::make(kernel_layout, kOIW);
  CHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = BijectiveLayoutNode::make(out_layout, kNCW);
  CHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_ncw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    Array<IndexExpr> wshape;

    wshape = {{param->channels, dshape_ncw[1], param->kernel_size[0]}};

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      // check the size
      CHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) )
          << "Conv1D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      CHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv1D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }
    CHECK(reporter->AssertEQ(dshape_ncw[1], wshape[1]));
    channels = wshape[0];
    dilated_ksize = 1 + (wshape[2] - 1) * param->dilation[0];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_ncw[0], channels, 0});

  if (!dshape_ncw[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(dshape_ncw[2] + param->padding[0] + param->padding[1] - dilated_ksize,
                           param->strides[0]) + 1);
  } else {
    oshape.Set(2, dshape_ncw[2]);
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

template <typename AttrType>
bool Conv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCHW("NCHW");
  static const Layout kOIHW("OIHW");

  const AttrType* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCHW);
  CHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = BijectiveLayoutNode::make(kernel_layout, kOIHW);
  CHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIHW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = BijectiveLayoutNode::make(out_layout, kNCHW);
  CHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_nchw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    CHECK_EQ(param->kernel_size.size(), 2);
    CHECK_EQ(param->dilation.size(), 2);
    Array<IndexExpr> wshape;

    if (tvm::tir::Equal(param->channels, param->groups) && !tvm::tir::Equal(param->channels, 1)) {
      // infer weight's shape for depthwise convolution
      wshape = {{dshape_nchw[1], indexdiv(param->groups, dshape_nchw[1]), param->kernel_size[0],
                 param->kernel_size[1]}};
    } else {
      wshape = {{param->channels, indexdiv(dshape_nchw[1], param->groups), param->kernel_size[0],
                 param->kernel_size[1]}};
    }

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize_y = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }
    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      CHECK_EQ(param->kernel_size.size(), 2);
      // check the size
      CHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
            reporter->AssertEQ(param->kernel_size[1], wshape[3]))
          << "Conv2D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      CHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv2D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }
    CHECK(reporter->AssertEQ(indexdiv(dshape_nchw[1], param->groups), wshape[1]));
    channels = wshape[0];
    dilated_ksize_y = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_x = 1 + (wshape[3] - 1) * param->dilation[1];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_nchw[0], channels, 0, 0});

  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(param->padding, &pad_h, &pad_w);
  if (!dshape_nchw[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(dshape_nchw[2] + pad_h - dilated_ksize_y,
                           param->strides[0]) + 1);
  } else {
    oshape.Set(2, dshape_nchw[2]);
  }

  if (!dshape_nchw[3].as<tir::AnyNode>()) {
    oshape.Set(3, indexdiv(dshape_nchw[3] + pad_w - dilated_ksize_x,
                           param->strides[1]) + 1);
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

template <typename AttrType>
bool Conv3DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr) return false;
  static const Layout kNCDHW("NCDHW");
  static const Layout kOIDHW("OIDHW");

  const AttrType* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  const Layout in_layout(param->data_layout);
  const Layout kernel_layout(param->kernel_layout);

  const auto trans_in_layout = BijectiveLayoutNode::make(in_layout, kNCDHW);
  CHECK(trans_in_layout.defined())
      << "Conv only support input layouts that are convertible from NCDHW."
      << " But got " << in_layout;

  const auto trans_kernel_layout = BijectiveLayoutNode::make(kernel_layout, kOIDHW);
  CHECK(trans_kernel_layout.defined())
      << "Conv only support kernel layouts that are convertible from OIDHW."
      << " But got " << kernel_layout;

  Layout out_layout(param->out_layout == "" ? param->data_layout : param->out_layout);
  const auto trans_out_layout = BijectiveLayoutNode::make(out_layout, kNCDHW);
  CHECK(trans_out_layout.defined())
      << "Conv only support output layouts that are convertible from NCDHW."
      << " But got " << out_layout;

  Array<IndexExpr> dshape_ncdhw = trans_in_layout.ForwardShape(data->shape);

  IndexExpr channels, dilated_ksize_z, dilated_ksize_y, dilated_ksize_x;
  // infer weight if the kernel_size and channels are defined
  if (param->kernel_size.defined() && param->channels.defined()) {
    CHECK_EQ(param->kernel_size.size(), 3);
    CHECK_EQ(param->dilation.size(), 3);
    Array<IndexExpr> wshape;

    if (tvm::tir::Equal(param->channels, param->groups) && !tvm::tir::Equal(param->channels, 1)) {
      // infer weight's shape for depthwise convolution
      wshape = {{dshape_ncdhw[1], indexdiv(param->groups, dshape_ncdhw[1]), param->kernel_size[0],
                 param->kernel_size[1], param->kernel_size[2]}};
    } else {
      wshape = {{param->channels, indexdiv(dshape_ncdhw[1], param->groups), param->kernel_size[0],
                 param->kernel_size[1], param->kernel_size[2]}};
    }

    wshape = trans_kernel_layout.BackwardShape(wshape);
    channels = param->channels;
    dilated_ksize_z = 1 + (param->kernel_size[0] - 1) * param->dilation[0];
    dilated_ksize_y = 1 + (param->kernel_size[1] - 1) * param->dilation[1];
    dilated_ksize_x = 1 + (param->kernel_size[2] - 1) * param->dilation[2];
    DataType weight_dtype = data->dtype;
    if (weight != nullptr) {
      weight_dtype = weight->dtype;
    }

    // assign result to reporter
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
  } else {
    // use weight to infer the conv shape.
    if (weight == nullptr) return false;
    auto wshape = trans_kernel_layout.ForwardShape(weight->shape);
    if (param->kernel_size.defined()) {
      CHECK_EQ(param->kernel_size.size(), 3);
      // check the size
      CHECK(reporter->AssertEQ(param->kernel_size[0], wshape[2]) &&
            reporter->AssertEQ(param->kernel_size[1], wshape[3]) &&
            reporter->AssertEQ(param->kernel_size[2], wshape[4]))
          << "Conv3D: shape of weight is inconsistent with kernel_size, "
          << " kernel_size=" << param->kernel_size << " wshape=" << wshape;
    }
    if (param->channels.defined()) {
      CHECK(reporter->AssertEQ(param->channels, wshape[0]))
          << "Conv3D: shape of weight is inconsistent with channels, "
          << " channels=" << param->channels << " wshape=" << wshape;
    }
    CHECK(reporter->AssertEQ(indexdiv(dshape_ncdhw[1], param->groups), wshape[1]));
    channels = wshape[0];
    dilated_ksize_z = 1 + (wshape[2] - 1) * param->dilation[0];
    dilated_ksize_y = 1 + (wshape[3] - 1) * param->dilation[1];
    dilated_ksize_x = 1 + (wshape[4] - 1) * param->dilation[2];
  }
  // dilation
  Array<IndexExpr> oshape({dshape_ncdhw[0], channels, 0, 0, 0});

  IndexExpr pad_d, pad_h, pad_w;
  GetPaddingDepthHeightWidth(param->padding, &pad_d, &pad_h, &pad_w);
  if (!dshape_ncdhw[2].as<tir::AnyNode>()) {
    oshape.Set(2, indexdiv(dshape_ncdhw[2] + pad_d - dilated_ksize_z,
                           param->strides[0]) + 1);
  } else {
    oshape.Set(2, dshape_ncdhw[2]);
  }

  if (!dshape_ncdhw[3].as<tir::AnyNode>()) {
    oshape.Set(3, indexdiv(dshape_ncdhw[3] + pad_h - dilated_ksize_y,
                           param->strides[1]) + 1);
  } else {
    oshape.Set(3, dshape_ncdhw[3]);
  }

  if (!dshape_ncdhw[4].as<tir::AnyNode>()) {
    oshape.Set(4, indexdiv(dshape_ncdhw[4] + pad_w - dilated_ksize_x,
                           param->strides[2]) + 1);
  } else {
    oshape.Set(4, dshape_ncdhw[4]);
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

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_CONVOLUTION_H_
