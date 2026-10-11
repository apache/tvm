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
 * \file tvm/relax/op/nn.h
 * \brief Attributes for neural network operators.
 */
#ifndef TVM_RELAX_OP_NN_H_
#define TVM_RELAX_OP_NN_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in Conv1d operator */
struct Conv1DAttrsNode : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv1DAttrs", Conv1DAttrsNode, AttrsNode);
};  // struct Conv1dAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DAttrsNode : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv2DAttrs", Conv2DAttrsNode, AttrsNode);
};  // struct Conv2dAttrs

/*! \brief Attributes used in Conv3d operator */
struct Conv3DAttrsNode : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv3DAttrs", Conv3DAttrsNode, AttrsNode);
};  // struct Conv3dAttrs

/*! \brief Attributes used in Conv1DTranspose operator */
struct Conv1DTransposeAttrsNode : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> output_padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv1DTransposeAttrs", Conv1DTransposeAttrsNode,
                                    AttrsNode);
};  // struct Conv1DTransposeAttrsNode

/*! \brief Attributes used in Conv2d operator */
struct Conv2DTransposeAttrsNode : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> output_padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv2DTransposeAttrs", Conv2DTransposeAttrsNode,
                                    AttrsNode);
};  // struct Conv2DTransposeAttrsNode

/*! \brief Attributes used in Conv3dTranspose operator */
struct Conv3DTransposeAttrsNode : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> output_padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv3DTransposeAttrs", Conv3DTransposeAttrsNode,
                                    AttrsNode);
};  // struct Conv3DTransposeAttrsNode

/*! \brief Attributes used in max_pool1d and avg_pool1d operator */
struct Pool1DAttrsNode : public AttrsNode {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool1DAttrs", Pool1DAttrsNode, AttrsNode);
};  // struct Pool1dAttrs

/*! \brief Attributes used in max_pool2d and avg_pool2d operator */
struct Pool2DAttrsNode : public AttrsNode {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool2DAttrs", Pool2DAttrsNode, AttrsNode);
};  // struct Pool2dAttrs

/*! \brief Attributes used in max_pool3d and avg_pool3d operator */
struct Pool3DAttrsNode : public AttrsNode {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool3DAttrs", Pool3DAttrsNode, AttrsNode);
};  // struct Pool3dAttrs

/*! \brief Attributes for 1d adaptive pool operator */
struct AdaptivePool1DAttrsNode : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool1DAttrs", AdaptivePool1DAttrsNode,
                                    AttrsNode);
};  // struct AdaptivePool1DAttrsNode

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrsNode : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool2DAttrs", AdaptivePool2DAttrsNode,
                                    AttrsNode);
};  // struct AdaptivePool2DAttrsNode

/*! \brief Attributes for 3d adaptive pool operator */
struct AdaptivePool3DAttrsNode : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool3DAttrs", AdaptivePool3DAttrsNode,
                                    AttrsNode);
};  // struct AdaptivePool3DAttrsNode

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrsNode : public AttrsNode {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SoftmaxAttrs", SoftmaxAttrsNode, AttrsNode);
};

/*! \brief Attributes used in softmax operators */
struct LeakyReluAttrsNode : public AttrsNode {
  double alpha;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LeakyReluAttrs", LeakyReluAttrsNode, AttrsNode);
};

/*! \brief Attributes used in softplus operators */
struct SoftplusAttrsNode : public AttrsNode {
  double beta;
  double threshold;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SoftplusAttrs", SoftplusAttrsNode, AttrsNode);
};

/*! \brief Attributes used in PReLU operator */
struct PReluAttrsNode : public AttrsNode {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PReluAttrs", PReluAttrsNode, AttrsNode);
};

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrsNode : public AttrsNode {
  int axis;
  double epsilon;
  bool center;
  bool scale;
  double momentum;
  bool training;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.BatchNormAttrs", BatchNormAttrsNode, AttrsNode);
};  // struct BatchNormAttrsNode

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrsNode : public AttrsNode {
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayerNormAttrs", LayerNormAttrsNode, AttrsNode);
};  // struct LayerNormAttrsNode

/*! \brief Attributes used in group_norm operator */
struct GroupNormAttrsNode : public AttrsNode {
  int num_groups;
  int channel_axis;
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GroupNormAttrs", GroupNormAttrsNode, AttrsNode);
};  // struct GroupNormAttrsNode

/*! \brief Attributes used in instance_norm operator */
struct InstanceNormAttrsNode : public AttrsNode {
  int channel_axis;
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.InstanceNormAttrs", InstanceNormAttrsNode,
                                    AttrsNode);
};  // struct InstanceNormAttrsNode

/*! \brief Attributes used in rms_norm operator */
struct RMSNormAttrsNode : public AttrsNode {
  ffi::Array<int64_t> axes;
  double epsilon;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RMSNormAttrs", RMSNormAttrsNode, AttrsNode);
};  // struct RMSNormAttrsNode

/*! \brief Attributes used in nll_loss operator */
struct NLLLossAttrsNode : public AttrsNode {
  ffi::String reduction;
  int ignore_index;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.NLLLossAttrs", NLLLossAttrsNode, AttrsNode);
};  // struct NLLLossAttrsNode

/*! \brief Attributes used in dropout operator */
struct DropoutAttrsNode : public AttrsNode {
  double rate;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.DropoutAttrs", DropoutAttrsNode, AttrsNode);
};  // struct DropoutAttrsNode

/*! \brief Attributes used in Attention operator */
struct AttentionAttrsNode : public AttrsNode {
  ffi::Optional<FloatImm> scale;
  ffi::Optional<ffi::String> causal_mask;
  ffi::Optional<IntImm> window_size;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AttentionAttrs", AttentionAttrsNode, AttrsNode);
};  // struct AttentionAttrsNode

/*! \brief Attributes used for the padding operator */
struct PadAttrsNode : public AttrsNode {
  ffi::Array<int64_t> pad_width;
  double pad_value = 0.0;
  tvm::ffi::String pad_mode;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PadAttrs", PadAttrsNode, AttrsNode);
};

/*! \brief Attributes used for the pixel shuffle operator */
struct PixelShuffleAttrsNode : public AttrsNode {
  int upscale_factor;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PixelShuffleAttrs", PixelShuffleAttrsNode,
                                    AttrsNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_NN_H_
