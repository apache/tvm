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
 * \file tvm/relax/attrs/nn.h
 * \brief Attributes for neural network operators.
 */
#ifndef TVM_RELAX_ATTRS_NN_H_
#define TVM_RELAX_ATTRS_NN_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in Conv1d operator */
struct Conv1DAttrs : public AttrsNodeReflAdapter<Conv1DAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv1DAttrs>()
        .def_ro("strides", &Conv1DAttrs::strides, "Specifies the strides of the convolution.")
        .def_ro("padding", &Conv1DAttrs::padding,
                "If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on both sides"
                "two int : padding width in the order of (left, right)")
        .def_ro("dilation", &Conv1DAttrs::dilation,
                "Specifies the dilation rate to use for dilated convolution.")
        .def_ro("groups", &Conv1DAttrs::groups,
                "Number of groups to split the input into for grouped convolution. The number of "
                "input and "
                "output channels should be divisible by the number of groups.")
        .def_ro("data_layout", &Conv1DAttrs::data_layout,
                "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel, width"
                "dimensions respectively. Convolution is applied on the 'W' dimensions.")
        .def_ro("kernel_layout", &Conv1DAttrs::kernel_layout,
                "Dimension ordering of weight. Can be 'OIW', 'IOW', etc."
                "'O', 'I', 'W' stands for num_filter, input_channel, and width"
                "dimensions respectively.")
        .def_ro("out_layout", &Conv1DAttrs::out_layout,
                "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel, and width"
                "dimensions respectively. Default to be same as input layout.")
        .def_ro("out_dtype", &Conv1DAttrs::out_dtype,
                "Output data type, set to explicit type under mixed precision setting");
  }

  static constexpr const char* _type_key = "relax.attrs.Conv1DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Conv1DAttrs, BaseAttrsNode);
};  // struct Conv1dAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DAttrs : public AttrsNodeReflAdapter<Conv2DAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv2DAttrs>()
        .def_ro("strides", &Conv2DAttrs::strides, "Specifies the strides of the convolution.")
        .def_ro("padding", &Conv2DAttrs::padding,
                "If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : bottom, right will use same padding as top, left"
                "four int : padding width in the order of (top, left, bottom, right)")
        .def_ro("dilation", &Conv2DAttrs::dilation,
                "Specifies the dilation rate to use for dilated convolution.")
        .def_ro("groups", &Conv2DAttrs::groups,
                "Number of groups to split the input into for grouped convolution. The number of "
                "input and "
                "output channels should be divisible by the number of groups.")
        .def_ro("data_layout", &Conv2DAttrs::data_layout,
                "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.")
        .def_ro("kernel_layout", &Conv2DAttrs::kernel_layout,
                "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
                "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
                "dimensions respectively.")
        .def_ro("out_layout", &Conv2DAttrs::out_layout,
                "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Default to be same as input layout.")
        .def_ro("out_dtype", &Conv2DAttrs::out_dtype,
                "Output data type, set to explicit type under mixed precision setting");
  }

  static constexpr const char* _type_key = "relax.attrs.Conv2DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Conv2DAttrs, BaseAttrsNode);
};  // struct Conv2dAttrs

/*! \brief Attributes used in Conv3d operator */
struct Conv3DAttrs : public AttrsNodeReflAdapter<Conv3DAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv3DAttrs>()
        .def_ro("strides", &Conv3DAttrs::strides, "Specifies the strides of the convolution.")
        .def_ro(
            "padding", &Conv3DAttrs::padding,
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (forward, back, top, left, bottom, right)")
        .def_ro("dilation", &Conv3DAttrs::dilation,
                "Specifies the dilation rate to use for dilated convolution.")
        .def_ro("groups", &Conv3DAttrs::groups,
                "Number of groups to split the input into for grouped convolution. The number of "
                "input and "
                "output channels should be divisible by the number of groups.")
        .def_ro("data_layout", &Conv3DAttrs::data_layout,
                "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
                "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
                "dimensions respectively. Convolution is applied on the 'D', 'H', and"
                "'W' dimensions.")
        .def_ro(
            "kernel_layout", &Conv3DAttrs::kernel_layout,
            "Dimension ordering of weight. Can be 'OIDHW', 'OIDHW16o16i', etc."
            "'O', 'I', 'D', 'H', 'W' stands for num_filter, input_channel, depth, height, and width"
            "dimensions respectively.")
        .def_ro("out_layout", &Conv3DAttrs::out_layout,
                "Dimension ordering of output. Can be 'NCDHW', 'NDHWC', etc."
                "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
                "dimensions respectively. Default to be same as input layout.")
        .def_ro("out_dtype", &Conv3DAttrs::out_dtype,
                "Output data type, set to explicit type under mixed precision setting");
  }

  static constexpr const char* _type_key = "relax.attrs.Conv3DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Conv3DAttrs, BaseAttrsNode);
};  // struct Conv3dAttrs

/*! \brief Attributes used in Conv1DTranspose operator */
struct Conv1DTransposeAttrs : public AttrsNodeReflAdapter<Conv1DTransposeAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> output_padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv1DTransposeAttrs>()
        .def_ro("strides", &Conv1DTransposeAttrs::strides,
                "Specifies the strides of the convolution.")
        .def_ro("padding", &Conv1DTransposeAttrs::padding,
                "If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on both sides"
                "two int : padding width in the order of (left, right)")
        .def_ro("output_padding", &Conv1DTransposeAttrs::output_padding,
                "Used to disambiguate the output shape.")
        .def_ro("dilation", &Conv1DTransposeAttrs::dilation,
                "Specifies the dilation rate to use for dilated convolution.")
        .def_ro("groups", &Conv1DTransposeAttrs::groups,
                "Number of groups to split the input into for grouped convolution. The number of "
                "input and "
                "output channels should be divisible by the number of groups.")
        .def_ro("data_layout", &Conv1DTransposeAttrs::data_layout,
                "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel, width"
                "dimensions respectively. Convolution is applied on the 'W' dimensions.")
        .def_ro("kernel_layout", &Conv1DTransposeAttrs::kernel_layout,
                "Dimension ordering of weight. Can be 'OIW', 'IOW', etc."
                "'O', 'I', 'W' stands for num_filter, input_channel, and width"
                "dimensions respectively.")
        .def_ro("out_layout", &Conv1DTransposeAttrs::out_layout,
                "Dimension ordering of output. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel, and width"
                "dimensions respectively. Default to be same as input layout.")
        .def_ro("out_dtype", &Conv1DTransposeAttrs::out_dtype,
                "Output data type, set to explicit type under mixed precision setting");
  }

  static constexpr const char* _type_key = "relax.attrs.Conv1DTransposeAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Conv1DTransposeAttrs, BaseAttrsNode);
};  // struct Conv1DTransposeAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DTransposeAttrs : public AttrsNodeReflAdapter<Conv2DTransposeAttrs> {
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> output_padding;
  Array<IntImm> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Conv2DTransposeAttrs>()
        .def_ro("strides", &Conv2DTransposeAttrs::strides,
                "Specifies the strides of the convolution.")
        .def_ro("padding", &Conv2DTransposeAttrs::padding,
                "If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : bottom, right will use same padding as top, left"
                "four int : padding width in the order of (top, left, bottom, right)")
        .def_ro("output_padding", &Conv2DTransposeAttrs::output_padding,
                "Used to disambiguate the output shape.")
        .def_ro("dilation", &Conv2DTransposeAttrs::dilation,
                "Specifies the dilation rate to use for dilated convolution.")
        .def_ro("groups", &Conv2DTransposeAttrs::groups,
                "Number of groups to split the input into for grouped convolution. The number of "
                "input and "
                "output channels should be divisible by the number of groups.")
        .def_ro("data_layout", &Conv2DTransposeAttrs::data_layout,
                "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.")
        .def_ro("kernel_layout", &Conv2DTransposeAttrs::kernel_layout,
                "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
                "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
                "dimensions respectively.")
        .def_ro("out_layout", &Conv2DTransposeAttrs::out_layout,
                "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Default to be same as input layout.")
        .def_ro("out_dtype", &Conv2DTransposeAttrs::out_dtype,
                "Output data type, set to explicit type under mixed precision setting");
  }

  static constexpr const char* _type_key = "relax.attrs.Conv2DTransposeAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Conv2DTransposeAttrs, BaseAttrsNode);
};  // struct Conv2DTransposeAttrs

/*! \brief Attributes used in max_pool1d and avg_pool1d operator */
struct Pool1DAttrs : public AttrsNodeReflAdapter<Pool1DAttrs> {
  Array<IntImm> pool_size;
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  bool ceil_mode;
  bool count_include_pad;
  String layout;
  String out_layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Pool1DAttrs>()
        .def_ro("pool_size", &Pool1DAttrs::pool_size, "Size of the pooling windows.")
        .def_ro("strides", &Pool1DAttrs::strides, "Specifies the strides of the convolution.")
        .def_ro("dilation", &Pool1DAttrs::dilation, "Specifies the dilation of the convolution.")
        .def_ro("padding", &Pool1DAttrs::padding,
                "If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : padding width in the order of (left, right)")
        .def_ro(
            "ceil_mode", &Pool1DAttrs::ceil_mode,
            "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
            "every element in the input tensor will be covered by a sliding window.")
        .def_ro("count_include_pad", &Pool1DAttrs::count_include_pad,
                "When true, will include padding to compute the average")
        .def_ro("layout", &Pool1DAttrs::layout,
                "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel, and width"
                "dimensions respectively. Pooling is applied on the 'W' dimensions.",
                refl::DefaultValue("NCW"))
        .def_ro("out_layout", &Pool1DAttrs::out_layout,
                "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel, and width"
                "dimensions respectively. Pooling is applied on the 'W' dimensions.");
  }

  static constexpr const char* _type_key = "relax.attrs.Pool1DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Pool1DAttrs, BaseAttrsNode);
};  // struct Pool1dAttrs

/*! \brief Attributes used in max_pool2d and avg_pool2d operator */
struct Pool2DAttrs : public AttrsNodeReflAdapter<Pool2DAttrs> {
  Array<IntImm> pool_size;
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  bool ceil_mode;
  bool count_include_pad;
  String layout;
  String out_layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Pool2DAttrs>()
        .def_ro("pool_size", &Pool2DAttrs::pool_size, "Size of the pooling windows.")
        .def_ro("strides", &Pool2DAttrs::strides, "Specifies the strides of the convolution.")
        .def_ro("dilation", &Pool2DAttrs::dilation, "Specifies the dilation of the convolution.")
        .def_ro("padding", &Pool2DAttrs::padding,
                "If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "two int : bottom, right will use same padding as top, left"
                "four int : padding width in the order of (top, left, bottom, right)")
        .def_ro(
            "ceil_mode", &Pool2DAttrs::ceil_mode,
            "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
            "every element in the input tensor will be covered by a sliding window.")
        .def_ro("count_include_pad", &Pool2DAttrs::count_include_pad,
                "When true, will include padding to compute the average")
        .def_ro("layout", &Pool2DAttrs::layout,
                "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Pooling is applied on the 'H' and"
                "'W' dimensions.")
        .def_ro("out_layout", &Pool2DAttrs::out_layout,
                "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Pooling is applied on the 'H' and"
                "'W' dimensions.");
  }

  static constexpr const char* _type_key = "relax.attrs.Pool2DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Pool2DAttrs, BaseAttrsNode);
};  // struct Pool2dAttrs

/*! \brief Attributes used in max_pool3d and avg_pool3d operator */
struct Pool3DAttrs : public AttrsNodeReflAdapter<Pool3DAttrs> {
  Array<IntImm> pool_size;
  Array<IntImm> strides;
  Array<IntImm> padding;
  Array<IntImm> dilation;
  bool ceil_mode;
  bool count_include_pad;
  String layout;
  String out_layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<Pool3DAttrs>()
        .def_ro("pool_size", &Pool3DAttrs::pool_size, "Size of the pooling windows.")
        .def_ro("strides", &Pool3DAttrs::strides, "Specifies the strides of the convolution.")
        .def_ro("dilation", &Pool3DAttrs::dilation, "Specifies the dilation of the convolution.")
        .def_ro("padding", &Pool3DAttrs::padding,
                "If padding is non-zero, then the input is implicitly zero-padded"
                "Padding support both symmetric and asymmetric as"
                "one int : same padding used on all sides"
                "three int : back, bottom, right will use same padding as front, top, left"
                "four int : padding width in the order of (front, top, left, back, bottom, right)")
        .def_ro(
            "ceil_mode", &Pool3DAttrs::ceil_mode,
            "A boolean indicating if use ceil or floor to compute the output shape. By using ceil, "
            "every element in the input tensor will be covered by a sliding window.")
        .def_ro("count_include_pad", &Pool3DAttrs::count_include_pad,
                "When true, will include padding to compute the average")
        .def_ro("layout", &Pool3DAttrs::layout,
                "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
                "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
                "dimensions respectively. Pooling is applied on the 'D', 'H' and"
                "'W' dimensions.")
        .def_ro("out_layout", &Pool3DAttrs::out_layout,
                "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
                "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
                "dimensions respectively. Pooling is applied on the 'D', 'H' and"
                "'W' dimensions.");
  }

  static constexpr const char* _type_key = "relax.attrs.Pool3DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(Pool3DAttrs, BaseAttrsNode);
};  // struct Pool3dAttrs

/*! \brief Attributes for 1d adaptive pool operator */
struct AdaptivePool1DAttrs : public AttrsNodeReflAdapter<AdaptivePool1DAttrs> {
  Optional<Array<IntImm>> output_size;
  String layout;
  String out_layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AdaptivePool1DAttrs>()
        .def_ro("output_size", &AdaptivePool1DAttrs::output_size, "Output width.")
        .def_ro("layout", &AdaptivePool1DAttrs::layout,
                "Dimension ordering of input data. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel and width"
                "dimensions respectively. Pooling is applied on the"
                "'W' dimensions.")
        .def_ro("out_layout", &AdaptivePool1DAttrs::out_layout,
                "Dimension ordering of output data. Can be 'NCW', 'NWC', etc."
                "'N', 'C', 'W' stands for batch, channel and width"
                "dimensions respectively. Pooling is applied on the"
                "'W' dimensions.");
  }

  static constexpr const char* _type_key = "relax.attrs.AdaptivePool1DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(AdaptivePool1DAttrs, BaseAttrsNode);
};  // struct AdaptivePool1DAttrs

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrs : public AttrsNodeReflAdapter<AdaptivePool2DAttrs> {
  Optional<Array<IntImm>> output_size;
  String layout;
  String out_layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AdaptivePool2DAttrs>()
        .def_ro("output_size", &AdaptivePool2DAttrs::output_size, "Output height and width.")
        .def_ro("layout", &AdaptivePool2DAttrs::layout,
                "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Pooling is applied on the 'H' and"
                "'W' dimensions.")
        .def_ro("out_layout", &AdaptivePool2DAttrs::out_layout,
                "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Pooling is applied on the 'H' and"
                "'W' dimensions.");
  }

  static constexpr const char* _type_key = "relax.attrs.AdaptivePool2DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(AdaptivePool2DAttrs, BaseAttrsNode);
};  // struct AdaptivePool2DAttrs

/*! \brief Attributes for 3d adaptive pool operator */
struct AdaptivePool3DAttrs : public AttrsNodeReflAdapter<AdaptivePool3DAttrs> {
  Optional<Array<IntImm>> output_size;
  String layout;
  String out_layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AdaptivePool3DAttrs>()
        .def_ro("output_size", &AdaptivePool3DAttrs::output_size, "Output depth, height and width.")
        .def_ro("layout", &AdaptivePool3DAttrs::layout,
                "Dimension ordering of input data. Can be 'NCDHW', 'NDHWC', etc."
                "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
                "dimensions respectively. Pooling is applied on 'D', 'H' and"
                "'W' dimensions.")
        .def_ro("out_layout", &AdaptivePool3DAttrs::out_layout,
                "Dimension ordering of output data. Can be 'NCDHW', 'NDHWC', etc."
                "'N', 'C', 'D', 'H', 'W' stands for batch, channel, depth, height, and width"
                "dimensions respectively. Pooling is applied on 'D', 'H' and"
                "'W' dimensions.");
  }

  static constexpr const char* _type_key = "relax.attrs.AdaptivePool3DAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(AdaptivePool3DAttrs, BaseAttrsNode);
};  // struct AdaptivePool3DAttrs

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public AttrsNodeReflAdapter<SoftmaxAttrs> {
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SoftmaxAttrs>().def_ro("axis", &SoftmaxAttrs::axis,
                                           "The axis to sum over when computing softmax.");
  }

  static constexpr const char* _type_key = "relax.attrs.SoftmaxAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(SoftmaxAttrs, BaseAttrsNode);
};

/*! \brief Attributes used in softmax operators */
struct LeakyReluAttrs : public AttrsNodeReflAdapter<LeakyReluAttrs> {
  double alpha;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LeakyReluAttrs>().def_ro("alpha", &LeakyReluAttrs::alpha,
                                             "The slope of the negative part.");
  }

  static constexpr const char* _type_key = "relax.attrs.LeakyReluAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(LeakyReluAttrs, BaseAttrsNode);
};

/*! \brief Attributes used in softplus operators */
struct SoftplusAttrs : public AttrsNodeReflAdapter<SoftplusAttrs> {
  double beta;
  double threshold;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SoftplusAttrs>()
        .def_ro("beta", &SoftplusAttrs::beta,
                "Scaling factor controlling the sharpness of the Softplus transition.")
        .def_ro("threshold", &SoftplusAttrs::threshold,
                "Value determining when to use linear approximation for numerical stability.");
  }

  static constexpr const char* _type_key = "relax.attrs.SoftplusAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(SoftplusAttrs, BaseAttrsNode);
};

/*! \brief Attributes used in PReLU operator */
struct PReluAttrs : public AttrsNodeReflAdapter<PReluAttrs> {
  int axis;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PReluAttrs>().def_ro("axis", &PReluAttrs::axis,
                                         "The axis along which the alpha values are applied.");
  }

  static constexpr const char* _type_key = "relax.attrs.PReluAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(PReluAttrs, BaseAttrsNode);
};

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public AttrsNodeReflAdapter<BatchNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;
  double momentum;
  bool training;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BatchNormAttrs>()
        .def_ro("axis", &BatchNormAttrs::axis, "The axis along which the normalization is applied.")
        .def_ro("epsilon", &BatchNormAttrs::epsilon,
                "Small float added to variance to avoid dividing by zero")
        .def_ro("center", &BatchNormAttrs::center,
                "Indicating if the beta offset will be added to the normalized tensor.")
        .def_ro("scale", &BatchNormAttrs::scale,
                "Indicating if the gamma scale will be multiplied.")
        .def_ro("momentum", &BatchNormAttrs::momentum,
                "The value used for the moving_mean and moving_var update.")
        .def_ro("training", &BatchNormAttrs::training,
                "Whether we are training (i.e., not in eval mode).");
  }

  static constexpr const char* _type_key = "relax.attrs.BatchNormAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(BatchNormAttrs, BaseAttrsNode);
};  // struct BatchNormAttrs

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public AttrsNodeReflAdapter<LayerNormAttrs> {
  Array<Integer> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayerNormAttrs>()
        .def_ro("axes", &LayerNormAttrs::axes,
                "The axes that along which the normalization is applied.")
        .def_ro("epsilon", &LayerNormAttrs::epsilon,
                "Small float added to variance to avoid dividing by zero")
        .def_ro("center", &LayerNormAttrs::center,
                "Indicating if the beta offset will be added to the normalized tensor.")
        .def_ro("scale", &LayerNormAttrs::scale,
                "Indicating if the gamma scale will be multiplied.");
  }

  static constexpr const char* _type_key = "relax.attrs.LayerNormAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(LayerNormAttrs, BaseAttrsNode);
};  // struct LayerNormAttrs

/*! \brief Attributes used in group_norm operator */
struct GroupNormAttrs : public AttrsNodeReflAdapter<GroupNormAttrs> {
  int num_groups;
  int channel_axis;
  Array<Integer> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GroupNormAttrs>()
        .def_ro("num_groups", &GroupNormAttrs::num_groups,
                "The number of groups to separate the channels into.")
        .def_ro("channel_axis", &GroupNormAttrs::channel_axis,
                "The axis that represents the channel.")
        .def_ro(
            "axes", &GroupNormAttrs::axes,
            "The axes that along which the normalization is applied (excluding the channel axis).")
        .def_ro("epsilon", &GroupNormAttrs::epsilon,
                "Small float added to variance to avoid dividing by zero")
        .def_ro("center", &GroupNormAttrs::center,
                "Indicating if the beta offset will be added to the normalized tensor.")
        .def_ro("scale", &GroupNormAttrs::scale,
                "Indicating if the gamma scale will be multiplied.");
  }

  static constexpr const char* _type_key = "relax.attrs.GroupNormAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(GroupNormAttrs, BaseAttrsNode);
};  // struct GroupNormAttrs

/*! \brief Attributes used in instance_norm operator */
struct InstanceNormAttrs : public AttrsNodeReflAdapter<InstanceNormAttrs> {
  int channel_axis;
  Array<Integer> axes;
  double epsilon;
  bool center;
  bool scale;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<InstanceNormAttrs>()
        .def_ro("channel_axis", &InstanceNormAttrs::channel_axis,
                "The axis that represents the channel.")
        .def_ro("axes", &InstanceNormAttrs::axes,
                "The axes that along which the normalization is applied.")
        .def_ro("epsilon", &InstanceNormAttrs::epsilon,
                "Small float added to variance to avoid dividing by zero")
        .def_ro("center", &InstanceNormAttrs::center,
                "Indicating if the beta offset will be added to the normalized tensor.")
        .def_ro("scale", &InstanceNormAttrs::scale,
                "Indicating if the gamma scale will be multiplied.");
  }

  static constexpr const char* _type_key = "relax.attrs.InstanceNormAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(InstanceNormAttrs, BaseAttrsNode);
};  // struct InstanceNormAttrs

/*! \brief Attributes used in rms_norm operator */
struct RMSNormAttrs : public AttrsNodeReflAdapter<RMSNormAttrs> {
  Array<Integer> axes;
  double epsilon;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RMSNormAttrs>()
        .def_ro("axes", &RMSNormAttrs::axes,
                "The axes that along which the normalization is applied.")
        .def_ro("epsilon", &RMSNormAttrs::epsilon,
                "Small float added to variance to avoid dividing by zero");
  }

  static constexpr const char* _type_key = "relax.attrs.RMSNormAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(RMSNormAttrs, BaseAttrsNode);
};  // struct RMSNormAttrs

/*! \brief Attributes used in nll_loss operator */
struct NLLLossAttrs : public AttrsNodeReflAdapter<NLLLossAttrs> {
  String reduction;
  int ignore_index;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NLLLossAttrs>()
        .def_ro("reduction", &NLLLossAttrs::reduction,
                "The reduction method to apply to the output. Can be"
                "'none', 'mean' or 'sum'.",
                refl::DefaultValue("mean"))
        .def_ro("ignore_index", &NLLLossAttrs::ignore_index, "The target value to ignore.");
  }

  static constexpr const char* _type_key = "relax.attrs.NLLLossAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(NLLLossAttrs, BaseAttrsNode);
};  // struct NLLLossAttrs

/*! \brief Attributes used in dropout operator */
struct DropoutAttrs : public AttrsNodeReflAdapter<DropoutAttrs> {
  double rate;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DropoutAttrs>().def_ro(
        "rate", &DropoutAttrs::rate,
        "Fraction of the input that gets dropped out during training time");
  }

  static constexpr const char* _type_key = "relax.attrs.DropoutAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(DropoutAttrs, BaseAttrsNode);
};  // struct DropoutAttrs

/*! \brief Attributes used in Attention operator */
struct AttentionAttrs : public AttrsNodeReflAdapter<AttentionAttrs> {
  Optional<FloatImm> scale;
  Optional<String> causal_mask;
  Optional<IntImm> window_size;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AttentionAttrs>()
        .def_ro(
            "scale", &AttentionAttrs::scale,
            "The custom scale applied before the softmax. The default value is 1 / sqrt(head_dim).")
        .def_ro("causal_mask", &AttentionAttrs::causal_mask,
                "The type of the causal mask, i.e. 'TopLeft' and 'BottomRight'.")
        .def_ro("window_size", &AttentionAttrs::window_size,
                "The size of the window for sliding-window attention.");
  }

  static constexpr const char* _type_key = "relax.attrs.AttentionAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(AttentionAttrs, BaseAttrsNode);
};  // struct AttentionAttrs

/*! \brief Attributes used for the padding operator */
struct PadAttrs : public AttrsNodeReflAdapter<PadAttrs> {
  Array<Integer> pad_width;
  double pad_value = 0.0;
  tvm::String pad_mode;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PadAttrs>()
        .def_ro("pad_width", &PadAttrs::pad_width,
                "Number of values padded to the edges of each axis, "
                "in the format of (before_1, after_1, ..., before_N, after_N)")
        .def_ro("pad_value", &PadAttrs::pad_value, "The value to fill in padded area with",
                refl::DefaultValue(0.0))
        .def_ro("pad_mode", &PadAttrs::pad_mode,
                "Padding type to use. \"constant\" pads with constant_value, "
                "\"edge\" pads using the edge values of the input array, "
                "\"reflect\" pads by reflecting values with respect to the edges.",
                refl::DefaultValue("constant"));
  }

  static constexpr const char* _type_key = "relax.attrs.PadAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(PadAttrs, BaseAttrsNode);
};

/*! \brief Attributes used for the pixel shuffle operator */
struct PixelShuffleAttrs : public AttrsNodeReflAdapter<PixelShuffleAttrs> {
  int upscale_factor;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PixelShuffleAttrs>().def_ro("upscale_factor",
                                                &PixelShuffleAttrs::upscale_factor,
                                                "Scale factor for spatial upsampling.");
  }

  static constexpr const char* _type_key = "relax.attrs.PixelShuffleAttrs";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(PixelShuffleAttrs, BaseAttrsNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_NN_H_
