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
 * \file tvm/relax/op.h
 * \brief Public attributes for Relax operator families.
 */
#ifndef TVM_RELAX_OP_H_
#define TVM_RELAX_OP_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/type.h>
#include <tvm/relax/distributed/global_info.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/global_info.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {

// Attributes for relax specific operators.

/*! \brief Attributes used in call_tir_with_grad */
struct CallTIRWithGradAttrs : public AttrsNode {
  ffi::String te_grad_name;
  ffi::Map<ffi::String, Any> te_grad_kwargs;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallTIRWithGradAttrs", CallTIRWithGradAttrs,
                                    AttrsNode);
};  // struct CallTIRAttrs

/*! \brief Attributes used in call_tir_inplace */
struct CallTIRInplaceAttrs : public AttrsNode {
  /*!
   * \brief Indices that describe which input corresponds to which output.
   *
   * If the `i`th member has the value `k` >= 0, then that means that input `k` should be used to
   * store the `i`th output. If an element has the value -1, that means a new tensor should be
   * allocated for that output.
   */
  ffi::Array<int64_t> inplace_indices;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallTIRInplaceAttrs", CallTIRInplaceAttrs,
                                    AttrsNode);
};  // struct CallTIRInplaceAttrs

/*! \brief Attributes used in call_inplace_packed */
struct CallInplacePackedAttrs : public AttrsNode {
  /*!
   * \brief Indices that describe which input corresponds to which output.
   *
   * If the `i`th member has the value `k` >= 0, then that means that input `k` should be used to
   * store the `i`th output. If an element has the value -1, that means the output will be newly
   * allocated.
   */
  ffi::Array<int64_t> inplace_indices;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.CallInplacePackedAttrs", CallInplacePackedAttrs,
                                    AttrsNode);
};  // struct CallInplacePackedAttrs

/*! \brief Attributes used in to_vdevice */
struct ToVDeviceAttrs : public AttrsNode {
  VDevice dst_vdevice;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ToVDeviceAttrs", ToVDeviceAttrs, AttrsNode);
};  // struct ToVDeviceAttrs

/*! \brief Attributes used in hint_on_device */
struct HintOnDeviceAttrs : public AttrsNode {
  int32_t device_type;
  int32_t index;
  MemoryScope memory_scope;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.HintOnDeviceAttrs", HintOnDeviceAttrs, AttrsNode);
};  // struct HintOnDeviceAttrs

// Attributes for tensor creation operators.

/*! \brief Attributes used in full/full_like, ones/ones_like, and zeros/zeros_like operators */
struct InitAttrs : public AttrsNode {
  ffi::Optional<DLDataType> dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.InitAttrs", InitAttrs, AttrsNode);
};  // struct InitAttrs

/*! \brief Attributes used in tril and triu operator */
struct TriluAttrs : public AttrsNode {
  int k;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TriluAttrs", TriluAttrs, AttrsNode);
};  // struct TriluAttrs

// Attributes for datatype operators.

/*! \brief Attributes used in astype operator */
struct AstypeAttrs : public AttrsNode {
  DLDataType dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AstypeAttrs", AstypeAttrs, AttrsNode);
};  // struct AstypeAttrs.

/*! \brief Attributes used in wrap_param operator */
struct WrapParamAttrs : public AttrsNode {
  DLDataType dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.WrapParamAttrs", WrapParamAttrs, AttrsNode);
};  // struct WrapParamAttrs.

// Attributes for indexing operators.

/*! \brief Attributes used in take operator */
struct TakeAttrs : public AttrsNode {
  ffi::Optional<int64_t> axis;
  ffi::String mode;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TakeAttrs", TakeAttrs, AttrsNode);
};  // struct TakeAttrs

/*! \brief Attributes used in strided_slice operator */
struct StridedSliceAttrs : public AttrsNode {
  bool assume_inbound;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StridedSliceAttrs", StridedSliceAttrs, AttrsNode);
};  // struct StridedSliceAttrs

// Attributes for linear algebra operators.

/*! \brief Attributes for matmul operator */
struct MatmulAttrs : public AttrsNode {
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MatmulAttrs", MatmulAttrs, AttrsNode);
};  // struct MatmulAttrs

/*! \brief Attributes used in einsum operator */
struct EinsumAttrs : public AttrsNode {
  ffi::String subscripts;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.EinsumAttrs", EinsumAttrs, AttrsNode);
};  // struct EinsumAttrs

// Attributes for tensor manipulation operators.

/*! \brief Attributes used in concat operators */
struct ConcatAttrs : public AttrsNode {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ConcatAttrs", ConcatAttrs, AttrsNode);
};  // struct ConcatAttrs

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public AttrsNode {
  ffi::Array<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ExpandDimsAttrs", ExpandDimsAttrs, AttrsNode);
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in layout_transform operator */
struct LayoutTransformAttrs : public AttrsNode {
  tirx::IndexMap index_map;
  // pad_value is chosen to be of PrimExpr type, as it represents constant TIR POD expression. This
  // needs to be revisited in case PrimExpr is evolved to represent symbolic expression in future.
  ffi::Optional<PrimExpr> pad_value;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayoutTransformAttrs", LayoutTransformAttrs,
                                    AttrsNode);
};  // struct LayoutTransformAttrs

/*! \brief Attributes used in permute_dims operator */
struct PermuteDimsAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> axes;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PermuteDimsAttrs", PermuteDimsAttrs, AttrsNode);
};  // struct PermuteDimsAttrs

/*! \brief Attributes used in split operator */
struct SplitAttrs : public AttrsNode {
  ffi::ObjectRef indices_or_sections;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SplitAttrs", SplitAttrs, AttrsNode);
};  // struct SplitAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SqueezeAttrs", SqueezeAttrs, AttrsNode);
};  // struct SqueezeAttrs

/*! \brief Attributes used in stack operators */
struct StackAttrs : public AttrsNode {
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StackAttrs", StackAttrs, AttrsNode);
};  // struct StackAttrs

/*! \brief Attributes used in repeat operators */
struct RepeatAttrs : public AttrsNode {
  int repeats;
  ffi::Optional<int64_t> axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RepeatAttrs", RepeatAttrs, AttrsNode);
};  // struct RepeatAttrs

/*! \brief Attributes used in tile operators */
struct TileAttrs : public AttrsNode {
  ffi::Array<int64_t> repeats;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TileAttrs", TileAttrs, AttrsNode);
};  // struct TileAttrs

/*! \brief Attributes used in flip operators */
struct FlipAttrs : public AttrsNode {
  int64_t axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.FlipAttrs", FlipAttrs, AttrsNode);
};  // struct FlipAttrs

/*! \brief Attributes used in reverse_sequence operators */
struct ReverseSequenceAttrs : public AttrsNode {
  int64_t seq_axis;
  int64_t batch_axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ReverseSequenceAttrs", ReverseSequenceAttrs,
                                    AttrsNode);
};  // struct ReverseSequenceAttrs

/*! \brief Attributes used in gather_elements operators */
struct GatherElementsAttrs : public AttrsNode {
  int64_t axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherElementsAttrs", GatherElementsAttrs,
                                    AttrsNode);
};  // struct GatherElementsAttrs

/*! \brief Attributes used in gather_nd operators */
struct GatherNDAttrs : public AttrsNode {
  int64_t batch_dims;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GatherNDAttrs", GatherNDAttrs, AttrsNode);
};  // struct GatherNDAttrs

/*! \brief Attributes used in index_put operator */
struct IndexPutAttrs : public AttrsNode {
  bool accumulate;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.IndexPutAttrs", IndexPutAttrs, AttrsNode);
};  // struct IndexPutAttrs

/*! \brief Attribute used in meshgrid operator */
struct MeshgridAttrs : public AttrsNode {
  ffi::Optional<ffi::String> indexing;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MeshgridAttrs", MeshgridAttrs, AttrsNode);
};

/*! \brief Attributes used in scatter_elements operators */
struct ScatterElementsAttrs : public AttrsNode {
  int64_t axis;
  ffi::String reduction;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterElementsAttrs", ScatterElementsAttrs,
                                    AttrsNode);
};  // struct ScatterElementsAttrs

/*! \brief Attributes used in scatter_nd operators */
struct ScatterNDAttrs : public AttrsNode {
  ffi::String reduction;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterNDAttrs", ScatterNDAttrs, AttrsNode);
};  // struct ScatterNDAttrs

/*! \brief Attributes used in slice_scatter operator */
struct SliceScatterAttrs : public AttrsNode {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SliceScatterAttrs", SliceScatterAttrs, AttrsNode);
};  // struct SliceScatterAttrs

/*! \brief Attributes used in one_hot operator */
struct OneHotAttrs : public AttrsNode {
  int depth;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.OneHotAttrs", OneHotAttrs, AttrsNode);
};  // struct OneHotAttrs

// Attributes for quantize/dequantize operators.

/*! \brief Attributes for relax.quantize/relax.dequantize operator */
struct QuantizeAttrs : public AttrsNode {
  DLDataType out_dtype;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.QuantizeAttrs", QuantizeAttrs, AttrsNode);
};  // QuantizeAttrs

// Attributes for sampling operators.

/*! \brief Attributes used in multinomial_from_uniform operator */
struct MultinomialFromUniformAttrs : public AttrsNode {
  DLDataType dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MultinomialFromUniformAttrs",
                                    MultinomialFromUniformAttrs, AttrsNode);
};  // struct MultinomialFromUniformAttrs

// Attributes for search operators.

/*! \brief Attributes for search operators */
struct ArgmaxArgminAttrs : public AttrsNode {
  ffi::Optional<int64_t> axis;
  bool keepdims;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ArgmaxArgminAttrs", ArgmaxArgminAttrs, AttrsNode);
};  // struct ArgmaxArgminAttrs

/*! \brief Attributes for bucketize operator */
struct BucketizeAttrs : public tvm::AttrsNode {
  bool out_int32;
  bool right;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.BucketizeAttrs", BucketizeAttrs, AttrsNode);
};  // struct BucketizeAttrs

// Attributes for sorting operators.

/*! \brief Attributes used in sort operator */
struct SortAttrs : public AttrsNode {
  int axis;
  bool descending;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SortAttrs", SortAttrs, AttrsNode);
};  // struct SortAttrs

/*! \brief Attributes used in argsort operator */
struct ArgsortAttrs : public AttrsNode {
  int axis;
  bool descending;
  ffi::Optional<DLDataType> dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ArgsortAttrs", ArgsortAttrs, AttrsNode);
};  // struct ArgsortAttrs

/*! \brief Attributes used in topk operator */
struct TopKAttrs : public AttrsNode {
  int k;
  int axis;
  bool largest;
  ffi::String ret_type;
  ffi::Optional<DLDataType> dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TopKAttrs", TopKAttrs, AttrsNode);
};  // struct TopKAttrs

// Attributes for statistical operators.

/*! \brief Attributes for statistical operators */
struct StatisticalAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> axis;
  bool keepdims;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.StatisticalAttrs", StatisticalAttrs, AttrsNode);
};  // struct StatisticalAttrs

/*! \brief Attributes used in scan operators like cumsum, cumprod */
struct ScanopAttrs : public AttrsNode {
  ffi::Optional<int64_t> axis;
  ffi::Optional<DLDataType> dtype;
  bool exclusive = false;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScanopAttrs", ScanopAttrs, AttrsNode);
};  // struct ScanopAttrs

// Attributes for neural network operators.

/*! \brief Attributes used in Conv1d operator */
struct Conv1DAttrs : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv1DAttrs", Conv1DAttrs, AttrsNode);
};  // struct Conv1dAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DAttrs : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv2DAttrs", Conv2DAttrs, AttrsNode);
};  // struct Conv2dAttrs

/*! \brief Attributes used in Conv3d operator */
struct Conv3DAttrs : public AttrsNode {
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  int groups;
  ffi::String data_layout;
  ffi::String kernel_layout;
  ffi::String out_layout;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv3DAttrs", Conv3DAttrs, AttrsNode);
};  // struct Conv3dAttrs

/*! \brief Attributes used in Conv1DTranspose operator */
struct Conv1DTransposeAttrs : public AttrsNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv1DTransposeAttrs", Conv1DTransposeAttrs,
                                    AttrsNode);
};  // struct Conv1DTransposeAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DTransposeAttrs : public AttrsNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv2DTransposeAttrs", Conv2DTransposeAttrs,
                                    AttrsNode);
};  // struct Conv2DTransposeAttrs

/*! \brief Attributes used in Conv3dTranspose operator */
struct Conv3DTransposeAttrs : public AttrsNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Conv3DTransposeAttrs", Conv3DTransposeAttrs,
                                    AttrsNode);
};  // struct Conv3DTransposeAttrs

/*! \brief Attributes used in max_pool1d and avg_pool1d operator */
struct Pool1DAttrs : public AttrsNode {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool1DAttrs", Pool1DAttrs, AttrsNode);
};  // struct Pool1dAttrs

/*! \brief Attributes used in max_pool2d and avg_pool2d operator */
struct Pool2DAttrs : public AttrsNode {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool2DAttrs", Pool2DAttrs, AttrsNode);
};  // struct Pool2dAttrs

/*! \brief Attributes used in max_pool3d and avg_pool3d operator */
struct Pool3DAttrs : public AttrsNode {
  ffi::Array<int64_t> pool_size;
  ffi::Array<int64_t> strides;
  ffi::Array<int64_t> padding;
  ffi::Array<int64_t> dilation;
  bool ceil_mode;
  bool count_include_pad;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Pool3DAttrs", Pool3DAttrs, AttrsNode);
};  // struct Pool3dAttrs

/*! \brief Attributes for 1d adaptive pool operator */
struct AdaptivePool1DAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool1DAttrs", AdaptivePool1DAttrs,
                                    AttrsNode);
};  // struct AdaptivePool1DAttrs

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool2DAttrs", AdaptivePool2DAttrs,
                                    AttrsNode);
};  // struct AdaptivePool2DAttrs

/*! \brief Attributes for 3d adaptive pool operator */
struct AdaptivePool3DAttrs : public AttrsNode {
  ffi::Optional<ffi::Array<int64_t>> output_size;
  ffi::String layout;
  ffi::String out_layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AdaptivePool3DAttrs", AdaptivePool3DAttrs,
                                    AttrsNode);
};  // struct AdaptivePool3DAttrs

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public AttrsNode {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SoftmaxAttrs", SoftmaxAttrs, AttrsNode);
};

/*! \brief Attributes used in softmax operators */
struct LeakyReluAttrs : public AttrsNode {
  double alpha;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LeakyReluAttrs", LeakyReluAttrs, AttrsNode);
};

/*! \brief Attributes used in softplus operators */
struct SoftplusAttrs : public AttrsNode {
  double beta;
  double threshold;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.SoftplusAttrs", SoftplusAttrs, AttrsNode);
};

/*! \brief Attributes used in PReLU operator */
struct PReluAttrs : public AttrsNode {
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PReluAttrs", PReluAttrs, AttrsNode);
};

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public AttrsNode {
  int axis;
  double epsilon;
  bool center;
  bool scale;
  double momentum;
  bool training;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.BatchNormAttrs", BatchNormAttrs, AttrsNode);
};  // struct BatchNormAttrs

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public AttrsNode {
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.LayerNormAttrs", LayerNormAttrs, AttrsNode);
};  // struct LayerNormAttrs

/*! \brief Attributes used in group_norm operator */
struct GroupNormAttrs : public AttrsNode {
  int num_groups;
  int channel_axis;
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GroupNormAttrs", GroupNormAttrs, AttrsNode);
};  // struct GroupNormAttrs

/*! \brief Attributes used in instance_norm operator */
struct InstanceNormAttrs : public AttrsNode {
  int channel_axis;
  ffi::Array<int64_t> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.InstanceNormAttrs", InstanceNormAttrs, AttrsNode);
};  // struct InstanceNormAttrs

/*! \brief Attributes used in rms_norm operator */
struct RMSNormAttrs : public AttrsNode {
  ffi::Array<int64_t> axes;
  double epsilon;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.RMSNormAttrs", RMSNormAttrs, AttrsNode);
};  // struct RMSNormAttrs

/*! \brief Attributes used in nll_loss operator */
struct NLLLossAttrs : public AttrsNode {
  ffi::String reduction;
  int ignore_index;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.NLLLossAttrs", NLLLossAttrs, AttrsNode);
};  // struct NLLLossAttrs

/*! \brief Attributes used in dropout operator */
struct DropoutAttrs : public AttrsNode {
  double rate;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.DropoutAttrs", DropoutAttrs, AttrsNode);
};  // struct DropoutAttrs

/*! \brief Attributes used in Attention operator */
struct AttentionAttrs : public AttrsNode {
  ffi::Optional<FloatImm> scale;
  ffi::Optional<ffi::String> causal_mask;
  ffi::Optional<IntImm> window_size;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AttentionAttrs", AttentionAttrs, AttrsNode);
};  // struct AttentionAttrs

/*! \brief Attributes used for the padding operator */
struct PadAttrs : public AttrsNode {
  ffi::Array<int64_t> pad_width;
  double pad_value = 0.0;
  tvm::ffi::String pad_mode;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PadAttrs", PadAttrs, AttrsNode);
};

/*! \brief Attributes used for the pixel shuffle operator */
struct PixelShuffleAttrs : public AttrsNode {
  int upscale_factor;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.PixelShuffleAttrs", PixelShuffleAttrs, AttrsNode);
};

// Attributes for image operators.

/*! \brief Attributes used in image resize2d operator */
struct Resize2DAttrs : public AttrsNode {
  ffi::Array<FloatImm> roi;
  ffi::String layout;
  ffi::String method;
  ffi::String coordinate_transformation_mode;
  ffi::String rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Resize2DAttrs", Resize2DAttrs, AttrsNode);
};  // struct Resize2dAttrs

/*! \brief Attributes used in image resize3d operator */
struct Resize3DAttrs : public AttrsNode {
  ffi::Array<FloatImm> roi;
  ffi::String layout;
  ffi::String method;
  ffi::String coordinate_transformation_mode;
  ffi::String rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.Resize3DAttrs", Resize3DAttrs, AttrsNode);
};  // struct Resize3DAttrs

/*! \brief Attributes used in image grid_sample operator */
struct GridSampleAttrs : public AttrsNode {
  ffi::String method;
  ffi::String layout;
  ffi::String padding_mode;
  bool align_corners;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GridSampleAttrs", GridSampleAttrs, AttrsNode);
};  // struct GridSampleAttrs

/*! \brief Attributes used in image affine_grid operator */
struct AffineGridAttrs : public AttrsNode {
  bool align_corners;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AffineGridAttrs", AffineGridAttrs, AttrsNode);
};  // struct AffineGridAttrs

// Auxiliary attributes for vision operators.

/*! \brief Attributes used in AllClassNonMaximumSuppression operator */
struct AllClassNonMaximumSuppressionAttrs : public AttrsNode {
  ffi::String output_format;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllClassNonMaximumSuppressionAttrs",
                                    AllClassNonMaximumSuppressionAttrs, AttrsNode);
};  // struct AllClassNonMaximumSuppressionAttrs

/*! \brief Attributes used in ROIAlign operator */
struct ROIAlignAttrs : public AttrsNode {
  ffi::Array<int64_t> pooled_size;
  double spatial_scale;
  int sample_ratio;
  bool aligned;
  ffi::String layout;
  ffi::String mode;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ROIAlignAttrs", ROIAlignAttrs, AttrsNode);
};  // struct ROIAlignAttrs

/*! \brief Attributes used in ROIPool operator */
struct ROIPoolAttrs : public AttrsNode {
  ffi::Array<int64_t> pooled_size;
  double spatial_scale;
  ffi::String layout;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ROIPoolAttrs", ROIPoolAttrs, AttrsNode);
};  // struct ROIPoolAttrs

/*! \brief Attributes used in GetValidCounts operator */
struct GetValidCountsAttrs : public AttrsNode {
  double score_threshold;
  int id_index;
  int score_index;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.GetValidCountsAttrs", GetValidCountsAttrs,
                                    AttrsNode);
};  // struct GetValidCountsAttrs

/*! \brief Attributes used in NonMaximumSuppression operator */
struct NonMaximumSuppressionAttrs : public AttrsNode {
  int max_output_size;
  double iou_threshold;
  bool force_suppress;
  int top_k;
  int coord_start;
  int score_index;
  int id_index;
  bool return_indices;
  bool invalid_to_bottom;
  double soft_nms_sigma;
  double score_threshold;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.NonMaximumSuppressionAttrs",
                                    NonMaximumSuppressionAttrs, AttrsNode);
};  // struct NonMaximumSuppressionAttrs

/*! \brief Attributes for multibox_transform_loc (SSD / TFLite-style box decode). */
struct MultiboxTransformLocAttrs : public AttrsNode {
  bool clip;
  double threshold;
  ffi::Array<double> variances;
  bool keep_background;
  bool apply_softmax;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MultiboxTransformLocAttrs",
                                    MultiboxTransformLocAttrs, AttrsNode);
};  // struct MultiboxTransformLocAttrs

// Attributes for ccl operators.

/*! \brief Attributes used in allreduce operators */
struct AllReduceAttrs : public tvm::AttrsNode {
  ffi::String op_type;
  bool in_group;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllReduceAttrs", AllReduceAttrs, AttrsNode);
};  // struct AllReduceAttrs

/*! \brief Attributes used in allgather operators */
struct AllGatherAttrs : public tvm::AttrsNode {
  int num_workers;
  bool in_group;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.AllGatherAttrs", AllGatherAttrs, AttrsNode);
};  // struct AllGatherAttrs

/*! \brief Attributes used in scatter operators */
struct ScatterCollectiveAttrs : public tvm::AttrsNode {
  int num_workers;
  int axis;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.ScatterCollectiveAttrs", ScatterCollectiveAttrs,
                                    AttrsNode);
};  // struct ScatterCollectiveAttrs

// Attributes for redistribute and annotate_sharding operators.

/*! \brief Attributes for redistribute and annotate_sharding operator */
struct DistributionAttrs : public AttrsNode {
  distributed::DeviceMesh device_mesh;
  distributed::Placement placement;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.DistributionAttrs", DistributionAttrs, AttrsNode);
};  // struct DistributionAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_H_
