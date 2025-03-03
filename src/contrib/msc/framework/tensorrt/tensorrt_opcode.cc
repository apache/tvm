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
 * \file src/contrib/msc/framework/tensorrt/tensorrt_opcode.cc
 */
#include "tensorrt_opcode.h"

#include <memory>
#include <string>

#include "../../core/utils.h"

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> TensorRTOpCode::GetDocs() {
  stack_.Config(this);
  CodeGenBuild();
  if (node()->optype == "tuple") {
    for (size_t i = 0; i < node()->outputs.size(); i++) {
      stack_.func_call("setName", NullOpt, DocUtils::ToPtr(IdxOutput(i)))
          .call_arg(DocUtils::ToStr(node()->OutputAt(i)->name));
    }
  } else if (node()->optype == "get_item") {
    stack_.func_call("setName", NullOpt, DocUtils::ToPtr(IdxNode()))
        .call_arg(DocUtils::ToStr(node()->OutputAt(0)->name));
  } else if (node()->optype != "input") {
    SetLayerByValue("Name", DocUtils::ToStr(node()->name));
    for (size_t i = 0; i < node()->outputs.size(); i++) {
      stack_.func_call("setName", NullOpt, DocUtils::ToPtr(IdxOutput(i)))
          .call_arg(DocUtils::ToStr(node()->OutputAt(i)->name));
    }
  }
  return stack_.GetDocs();
}

void TensorRTOpCode::SetPadding(const String& key) {
  const auto& padding = node()->GetTypeArrayAttr<int>("padding");
  if (padding.size() == 1) {
    SetLayerByDimsValue("Padding", std::vector<int>{padding[0], padding[0]}, false);
  } else if (padding.size() == 2) {
    SetLayerByDimsValue("PrePadding", padding, false);
    SetLayerByDimsValue("PostPadding", padding, false);
  } else if (padding.size() == 4) {
    SetLayerByDimsValue("PrePadding", std::vector<int>{padding[0], padding[1]}, false);
    SetLayerByDimsValue("PostPadding", std::vector<int>{padding[2], padding[3]}, false);
  } else {
    LOG_FATAL << "Unexpected padding size" << padding.size();
  }
}

const String TensorRTOpCode::DeclareInputs(bool simplify) {
  const String& inputs_ref = "inputs_" + std::to_string(node()->index);
  if (node()->parents.size() == 1 && simplify) {
    const auto& idx_input = StringUtils::Replace(IdxInput(), "*", "");
    stack_.declare("std::vector<ITensor*>", inputs_ref + "_vec")
        .declare_arg(node()->inputs.size())
        .declare_arg(idx_input);
  } else {
    stack_.declare("std::vector<ITensor*>", inputs_ref + "_vec", 0, false);
    for (size_t i = 0; i < node()->inputs.size(); i++) {
      const auto& idx_input = StringUtils::Replace(IdxInput(i), "*", "");
      stack_.declare_arg(idx_input);
    }
  }
  stack_.assign(inputs_ref, inputs_ref + "_vec.data()", "ITensor**");
  return inputs_ref;
}

const String TensorRTOpCode::DType(const DataType& dtype) {
  const String& dtype_name = BaseOpCode<TensorRTCodeGenConfig, TensorRTCodeGenHelper>::DType(dtype);
  String dtype_enum;
  if (dtype_name == "int8") {
    dtype_enum = "DataType::kINT8";
  } else if (dtype_name == "int32") {
    dtype_enum = "DataType::kINT32";
  } else if (dtype_name == "int64") {
    dtype_enum = "DataType::kINT32";
  } else if (dtype_name == "float16") {
    dtype_enum = "DataType::kHALF";
  } else if (dtype_name == "float32") {
    dtype_enum = "DataType::kFLOAT";
  } else {
    LOG_FATAL << "Unexpected dtype for TensorRT " << dtype_name;
  }
  return dtype_enum;
}

template <typename T>
const String TensorRTOpCode::ToDims(const std::vector<T>& dims, bool use_ndim) {
  if (dims.size() == 2 && !use_ndim) {
    return "DimsHW{" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "}";
  }
  String dims_str = "Dims({" + std::to_string(dims.size()) + ",{";
  for (size_t i = 0; i < dims.size(); i++) {
    dims_str = dims_str + std::to_string(dims[i]) + (i < dims.size() - 1 ? "," : "");
  }
  dims_str = dims_str + "}})";
  return dims_str;
}

const String TensorRTOpCode::ToDims(const Array<Integer>& dims, bool use_ndim) {
  std::vector<int64_t> int_dims;
  for (const auto& d : dims) {
    int_dims.push_back(d->value);
  }
  return ToDims(int_dims, use_ndim);
}

const String TensorRTOpCode::AttrToDims(const String& key, bool use_ndim) {
  const auto& dims = node()->GetTypeArrayAttr<int>(key);
  return ToDims(dims, use_ndim);
}

const size_t TensorRTOpCode::ToReduceAxis(const std::vector<int>& axes, size_t ndim) {
  size_t valid_ndim = ndim == 0 ? node()->InputAt(0)->Ndim() : ndim;
  size_t reduce_axis = 0;
  for (const auto& a : axes) {
    reduce_axis += 1 << CommonUtils::GetIndex(a, valid_ndim);
  }
  return reduce_axis;
}

const size_t TensorRTOpCode::AttrToReduceAxis(const String& key, size_t ndim) {
  std::vector<int> axes;
  if (node()->GetAttr(key, &axes)) {
    return ToReduceAxis(axes, ndim);
  }
  int axis;
  ICHECK(node()->GetAttr(key, &axis)) << "Can not get axes from attribute key " << key;
  return ToReduceAxis(std::vector<int>{axis}, ndim);
}

const size_t TensorRTOpCode::AttrToAxis(const String& key, size_t ndim) {
  size_t valid_ndim = ndim == 0 ? node()->InputAt(0)->Ndim() : ndim;
  int axis = node()->GetTypeAttr<int>(key);
  return CommonUtils::GetIndex(axis, valid_ndim);
}

template <typename T>
void TensorRTOpCode::SetLayerByAttr(const String& method, const String& key) {
  stack_.func_call("set" + method, NullOpt, DocUtils::ToPtr(IdxNode())).op_arg<T>(key, "");
}

template <typename T>
void TensorRTOpCode::SetLayerByValue(const String& method, const T& value) {
  stack_.func_call("set" + method, NullOpt, DocUtils::ToPtr(IdxNode())).call_arg(value);
}

void TensorRTOpCode::SetLayerByDimsAttr(const String& method, const String& key, bool use_ndim) {
  stack_.func_call("set" + method, NullOpt, DocUtils::ToPtr(IdxNode()))
      .call_arg(AttrToDims(key, use_ndim));
}

template <typename T>
void TensorRTOpCode::SetLayerByDimsValue(const String& method, const std::vector<T>& value,
                                         bool use_ndim) {
  stack_.func_call("set" + method, NullOpt, DocUtils::ToPtr(IdxNode()))
      .call_arg(ToDims(value, use_ndim));
}

void TensorRTOpCode::SetLayerByDimsValue(const String& method, const Array<Integer>& value,
                                         bool use_ndim) {
  stack_.func_call("set" + method, NullOpt, DocUtils::ToPtr(IdxNode()))
      .call_arg(ToDims(value, use_ndim));
}

#define TENSORRT_OP_CODEGEN_METHODS(TypeName) \
 public:                                      \
  TypeName(const String& func_name) : TensorRTOpCode(func_name) {}

#define TENSORRT_FLAG_OP_CODEGEN_METHODS(TypeName)                                      \
 public:                                                                                \
  TypeName(const String& func_name, const String& symbol) : TensorRTOpCode(func_name) { \
    symbol_ = symbol;                                                                   \
  }                                                                                     \
                                                                                        \
 private:                                                                               \
  String symbol_;

class TensorRTActivationCodeGen : public TensorRTOpCode {
 public:
  explicit TensorRTActivationCodeGen(const String& symbol) : TensorRTOpCode("Activation") {
    symbol_ = symbol;
  }

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().call_arg("ActivationType::k" + symbol_);
    if (node()->optype == "nn.leaky_relu") {
      SetLayerByAttr<float>("Alpha", "alpha");
    } else if (node()->optype == "clip") {
      SetLayerByAttr<float>("Alpha", "min");
      SetLayerByAttr<float>("Beta", "max");
    }
  }

 private:
  String symbol_;
};

class TensorRTAdaptivePool2dCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_FLAG_OP_CODEGEN_METHODS(TensorRTAdaptivePool2dCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& input = node()->InputAt(0);
    const auto& output = node()->OutputAt(0);
    std::vector<int64_t> in_sizes{input->DimAt("H")->value, input->DimAt("W")->value};
    std::vector<int64_t> out_sizes{output->DimAt("H")->value, output->DimAt("W")->value};
    std::vector<int64_t> stride, kernel;
    for (size_t i = 0; i < 2; i++) {
      stride.push_back(in_sizes[i] / out_sizes[i]);
      kernel.push_back((in_sizes[i] - (out_sizes[i] - 1) * stride[i]));
    }
    const String& suffix = CompareVersion(8, 0, 0) >= 0 ? "Nd" : "";
    stack_.op_call()
        .op_input_arg()
        .call_arg("PoolingType::k" + symbol_)
        .call_arg(ToDims(kernel, false));
    SetLayerByDimsValue("Stride" + suffix, stride, false);
  }
};

class TensorRTArgmaxminCodeGen : public TensorRTOpCode {
 public:
  explicit TensorRTArgmaxminCodeGen(const String& symbol) : TensorRTOpCode("TopK") {
    symbol_ = symbol;
  }

 protected:
  void CodeGenBuild() final {
    ICHECK(node()->GetTypeAttr<bool>("keepdims")) << "Only support argsort with keepdims";
    stack_.op_call()
        .op_input_arg()
        .call_arg("TopKOperation::k" + symbol_)
        .op_arg<bool>("keepdims", "")
        .call_arg(AttrToReduceAxis());
  }

 private:
  String symbol_;
};

class TensorRTAstypeCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTAstypeCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .func_call("setOutputType", NullOpt, DocUtils::ToPtr(IdxNode()))
        .call_arg(0)
        .op_dtype_arg(node()->OutputAt(0)->dtype);
  }
};

class TensorRTBatchMatmulCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTBatchMatmulCodeGen)

 protected:
  void CodeGenBuild() final {
    bool trans_a = node()->GetTypeAttr<bool>("transpose_a");
    bool trans_b = node()->GetTypeAttr<bool>("transpose_b");
    stack_.op_call()
        .op_input_arg()
        .call_arg(trans_a ? "MatrixOperation::kTRANSPOSE" : "MatrixOperation::kNONE")
        .op_input_arg(1)
        .call_arg(trans_b ? "MatrixOperation::kTRANSPOSE" : "MatrixOperation::kNONE");
  }
};

class TensorRTConcatCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTConcatCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& producer = node()->ProducerOf(0);
    ICHECK(node()->parents.size() == 1 && producer->optype == "tuple")
        << "Concat expect parent as tuple, get " << node();
    stack_.op_call().call_arg(IdxNodeBase(producer)).call_arg(producer->inputs.size());
    SetLayerByValue("Axis", AttrToAxis());
  }
};

class TensorRTConstantCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTConstantCodeGen)

 protected:
  void CodeGenBuild() final {
    ICHECK(!node()->HasAttr("scalar")) << "Scalar constant is not supported";
    stack_.op_call().call_arg(ToDims(node()->OutputAt(0)->shape)).op_weight_arg("const");
  }
};

class TensorRTConvCodeGen : public TensorRTOpCode {
 public:
  TensorRTConvCodeGen(const String& func_name, bool use_bias) : TensorRTOpCode(func_name) {
    use_bias_ = use_bias;
  }

 protected:
  void CodeGenBuild() final {
    const auto& weight = node()->WeightAt("weight");
    std::vector<int64_t> kernel_size;
    for (size_t i = 0; i < weight->Ndim(); i++) {
      if (weight->layout[i].name() == "I" || weight->layout[i].name() == "O") {
        continue;
      }
      kernel_size.push_back(weight->DimAt(i)->value);
    }
    stack_.op_call()
        .op_input_arg()
        .call_arg(weight->DimAt("O"))
        .call_arg(ToDims(kernel_size, false))
        .op_weight_arg("weight");
    if (use_bias_) {
      stack_.op_weight_arg("bias");
    } else {
      stack_.call_arg("mWeights[\"" + node()->name + ".bias\"]");
    }
    const String& suffix = CompareVersion(8, 0, 0) >= 0 ? "Nd" : "";
    SetLayerByDimsAttr("Stride" + suffix, "strides", false);
    SetLayerByDimsAttr("Dilation" + suffix, "dilation", false);
    SetLayerByAttr<int>("NbGroups", "groups");
    SetPadding();
  }

 private:
  bool use_bias_;
};

class TensorRTElemwiseCodeGen : public TensorRTOpCode {
 public:
  explicit TensorRTElemwiseCodeGen(const String& symbol) : TensorRTOpCode("ElementWise") {
    symbol_ = symbol;
  }

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_inputs_arg(false).call_arg("ElementWiseOperation::k" + symbol_);
  }

 private:
  String symbol_;
};

class TensorRTGetItemCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTGetItemCodeGen)

 protected:
  void CodeGenBuild() final {
    int index = node()->GetTypeAttr<int>("index");
    const auto& producer = node()->ProducerOf(0);
    stack_.assign(IdxNode(), IdxOutputBase(producer, index), "auto");
  }
};

class TensorRTInputCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTInputCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& output = node()->OutputAt(0);
    stack_.op_call()
        .call_arg(DocUtils::ToStr(output->name))
        .op_dtype_arg(output->dtype)
        .call_arg(ToDims(output->shape));
  }
};

class TensorRTLinearCodeGen : public TensorRTOpCode {
 public:
  TensorRTLinearCodeGen(const String& func_name, bool use_bias) : TensorRTOpCode(func_name) {
    use_bias_ = use_bias;
  }

 protected:
  void CodeGenBuild() final {
    const auto& weight = node()->WeightAt("weight");
    stack_.op_call().op_input_arg().call_arg(weight->DimAt("O")).op_weight_arg("weight");
    if (use_bias_) {
      stack_.op_weight_arg("bias");
    } else {
      stack_.call_arg(DocUtils::ToIndex("mWeights", DocUtils::ToStr(node()->name + ".bias")));
    }
  }

 private:
  bool use_bias_;
};

class TensorRTMatmulCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTMatmulCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .call_arg("MatrixOperation::kNONE")
        .op_input_arg(1)
        .call_arg("MatrixOperation::kNONE");
  }
};

class TensorRTPadCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTPadCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& pad_width = node()->GetTypeArrayAttr<int>("pad_width");
    ICHECK(pad_width.size() % 2 == 0) << "pad_width should be multiple of 2, get " << node();
    std::vector<int> pre_padding{2, 0}, post_padding{2, 0};
    const auto& input = node()->InputAt(0);
    for (size_t i = 0; i < input->Ndim(); i++) {
      if (input->layout[i].name() == "H") {
        pre_padding[0] = pad_width[i * 2];
        post_padding[0] = pad_width[i * 2 + 1];
      } else if (input->layout[i].name() == "W") {
        pre_padding[1] = pad_width[i * 2];
        post_padding[1] = pad_width[i * 2 + 1];
      }
    }
    stack_.op_call().op_input_arg().call_arg(ToDims(pre_padding)).call_arg(ToDims(post_padding));
  }
};

class TensorRTPermuteDimsCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTPermuteDimsCodeGen)

 protected:
  void CodeGenBuild() final {
    std::vector<int> axes;
    if (!node()->GetAttr("axes", &axes)) {
      for (size_t i = node()->InputAt(0)->Ndim(); i > 0; i--) {
        axes.push_back(i - 1);
      }
    }
    const String& perm_ref = "perm_" + std::to_string(node()->index);
    stack_.op_call().op_input_arg().declare("Permutation", perm_ref);
    for (size_t i = 0; i < axes.size(); i++) {
      stack_.assign(perm_ref + ".order[" + std::to_string(i) + "]",
                    CommonUtils::GetIndex(axes[i], node()->InputAt(0)->Ndim()));
    }
    SetLayerByValue("FirstTranspose", perm_ref);
  }
};

class TensorRTPool2dCodeGen : public TensorRTOpCode {
 public:
  explicit TensorRTPool2dCodeGen(const String& symbol) : TensorRTOpCode("PoolingNd") {
    symbol_ = symbol;
  }

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .call_arg("PoolingType::k" + symbol_)
        .call_arg(AttrToDims("pool_size", false));
    const String& suffix = CompareVersion(8, 0, 0) >= 0 ? "Nd" : "";
    SetLayerByDimsAttr("Stride" + suffix, "strides", false);
    if (node()->GetTypeAttr<bool>("ceil_mode")) {
      SetLayerByValue("PaddingMode", "PaddingMode::kEXPLICIT_ROUND_UP");
    }
    if (node()->optype == "nn.avg_pool2d") {
      SetLayerByValue("AverageCountExcludesPadding", false);
    }
    SetPadding();
  }

 private:
  String symbol_;
};

class TensorRTReduceCodeGen : public TensorRTOpCode {
 public:
  explicit TensorRTReduceCodeGen(const String& symbol) : TensorRTOpCode("Reduce") {
    symbol_ = symbol;
  }

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .call_arg("ReduceOperation::k" + symbol_)
        .call_arg(AttrToReduceAxis())
        .op_arg<bool>("keepdims", "");
  }

 private:
  String symbol_;
};

class TensorRTReshapeCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTReshapeCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& output = node()->OutputAt(0);
    stack_.op_call().op_input_arg();
    SetLayerByDimsValue("ReshapeDimensions", output->shape);
  }
};

class TensorRTResize2dCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTResize2dCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg();
    const auto& method = node()->GetTypeAttr<std::string>("method");
    String resize_mode;
    if (method == "linear") {
      resize_mode = "LINEAR";
    } else if (method == "nearest_neighbor") {
      resize_mode = "NEAREST";
    } else {
      LOG_FATAL << "Unexpected resize method " << method;
    }
    SetLayerByValue("ResizeMode", "ResizeMode::k" + resize_mode);
    SetLayerByValue("SelectorForSinglePixel", "ResizeSelector::kFORMULA");
    const auto& transformation_mode =
        node()->GetTypeAttr<std::string>("coordinate_transformation_mode");
    // set transformation
    if (transformation_mode == "align_corners") {
      SetLayerByValue("CoordinateTransformation", "ResizeCoordinateTransformation::kALIGN_CORNERS");
    } else if (transformation_mode == "asymmetric") {
      SetLayerByValue("CoordinateTransformation", "ResizeCoordinateTransformation::kASYMMETRIC");
    } else if (transformation_mode == "tf_half_pixel_for_nn") {
      SetLayerByValue("CoordinateTransformation", "ResizeCoordinateTransformation::kHALF_PIXEL");
    } else if (transformation_mode == "pytorch_half_pixel") {
      SetLayerByValue("CoordinateTransformation", "ResizeCoordinateTransformation::kHALF_PIXEL");
    } else if (transformation_mode == "half_pixel") {
      SetLayerByValue("CoordinateTransformation", "ResizeCoordinateTransformation::kHALF_PIXEL");
    } else {
      LOG_FATAL << "Unexpected transformation_mode " << transformation_mode;
    }
    // set round
    const auto& rounding_method = node()->GetTypeAttr<std::string>("rounding_method");
    if (transformation_mode == "tf_half_pixel_for_nn") {
      SetLayerByValue("NearestRounding", "ResizeRoundMode::kCEIL");
    } else if (rounding_method == "floor") {
      SetLayerByValue("NearestRounding", "ResizeRoundMode::kFLOOR");
    } else if (rounding_method == "ceil") {
      SetLayerByValue("NearestRounding", "ResizeRoundMode::kCEIL");
    } else if (rounding_method == "round_prefer_floor") {
      SetLayerByValue("NearestRounding", "ResizeRoundMode::kHALF_DOWN");
    } else if (rounding_method == "round_prefer_ceil") {
      SetLayerByValue("NearestRounding", "ResizeRoundMode::kHALF_UP");
    } else if (rounding_method == "round") {
      SetLayerByValue("NearestRounding", "ResizeRoundMode::kHALF_UP");
    } else if (rounding_method == "") {
      SetLayerByValue("NearestRounding", "ResizeRoundMode::kHALF_UP");
    } else {
      LOG_FATAL << "Unexpected rounding_method " << rounding_method;
    }
    // set output dims
    SetLayerByDimsValue("OutputDimensions", node()->OutputAt(0)->shape);
  }
};

class TensorRTSoftmaxCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTSoftmaxCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg();
    SetLayerByValue("Axes", AttrToReduceAxis());
  }
};

class TensorRTSquareCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTSquareCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_input_arg().call_arg("ElementWiseOperation::kPROD");
  }
};

class TensorRTStridedSliceCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTStridedSliceCodeGen)

 protected:
  void CodeGenBuild() final {
    std::vector<int> axes;
    if (!node()->GetAttr("axes", &axes)) {
      for (size_t i = 0; i < node()->InputAt(0)->Ndim(); i++) {
        axes.push_back(i);
      }
    }
    std::vector<size_t> begin(node()->InputAt(0)->Ndim(), 0);
    std::vector<size_t> strides(node()->InputAt(0)->Ndim(), 1);
    const auto& attr_begin = node()->GetTypeArrayAttr<int>("begin");
    for (size_t i = 0; i < axes.size(); i++) {
      size_t max_dim = static_cast<size_t>(node()->InputAt(0)->DimAt(axes[i])->value);
      begin[axes[i]] = CommonUtils::GetIndex(attr_begin[i], max_dim);
    }
    std::vector<int> attr_strides;
    if (node()->GetAttr("strides", &attr_strides)) {
      for (size_t i = 0; i < axes.size(); i++) {
        strides[axes[i]] = static_cast<size_t>(attr_strides[i]);
      }
    }
    stack_.op_call()
        .op_input_arg()
        .call_arg(ToDims(begin))
        .call_arg(ToDims(node()->OutputAt(0)->shape))
        .call_arg(ToDims(strides));
  }
};

class TensorRTTakeCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTTakeCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_inputs_arg(false).call_arg(AttrToAxis());
    if (node()->InputAt(0)->Ndim() == node()->InputAt(1)->Ndim()) {
      SetLayerByValue("Mode", "GatherMode::kELEMENT");
    }
  }
};

class TensorRTTopkCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTTopkCodeGen)

 protected:
  void CodeGenBuild() final {
    const String& symbol = node()->GetTypeAttr<bool>("largest") ? "MAX" : "MIN";
    stack_.op_call()
        .op_input_arg()
        .call_arg("TopKOperation::k" + symbol)
        .op_arg<int>("k", "")
        .call_arg(AttrToReduceAxis());
  }
};

class TensorRTTupleCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTTupleCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& inputs_ref = DeclareInputs();
    stack_.assign(IdxNode(), inputs_ref, "auto");
  }
};

class TensorRTUnaryCodeGen : public TensorRTOpCode {
 public:
  explicit TensorRTUnaryCodeGen(const String& symbol) : TensorRTOpCode("Unary") {
    symbol_ = symbol;
  }

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().call_arg("UnaryOperation::k" + symbol_);
  }

 private:
  String symbol_;
};

class TensorRTWhereCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTWhereCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg(false); }
};

class TensorRTPluginOpCodeGen : public TensorRTOpCode {
 public:
  TENSORRT_OP_CODEGEN_METHODS(TensorRTPluginOpCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& producer = node()->ParentAt(0);
    ICHECK(producer->optype == "tuple")
        << "Only support tensorrt plugin with tuple, get " << producer;

    const auto& plugin = GetPlugin(node()->optype);
    const auto& input_ref = "inputs_" + std::to_string(producer->index);
    const String& func_name = "plugin::" + node()->optype + "DynamicPlugin";
    const String& plugin_ref = "plugin_" + std::to_string(node()->index);
    const String& layouts_ref = "layouts_" + std::to_string(node()->index);
    stack_.declare("std::vector<std::string>", layouts_ref, 0, false);
    for (const auto& i : node()->GetInputs()) {
      stack_.declare_arg(DocUtils::ToStr(i->layout.name()));
    }
    stack_.func_call(func_name, DocUtils::ToDeclare("auto", plugin_ref))
        .call_arg(DocUtils::ToStr(node()->name));
    for (const auto& a : plugin->attrs) {
      stack_.call_arg(GetAttrDoc(a->name, a->type));
    }
    stack_.call_arg(layouts_ref);
    stack_.op_call().call_arg(input_ref).call_arg(plugin->inputs.size()).call_arg(plugin_ref);
  }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TensorRTOpCode>>>
GetTensorRTOpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<TensorRTOpCode>>>();
  if (!map->empty()) return map;
  // unary ops
  map->emplace("abs", std::make_shared<TensorRTUnaryCodeGen>("ABS"));
  map->emplace("acos", std::make_shared<TensorRTUnaryCodeGen>("ACOS"));
  map->emplace("acosh", std::make_shared<TensorRTUnaryCodeGen>("ACOSH"));
  map->emplace("asin", std::make_shared<TensorRTUnaryCodeGen>("ASIN"));
  map->emplace("asinh", std::make_shared<TensorRTUnaryCodeGen>("ASINH"));
  map->emplace("atan", std::make_shared<TensorRTUnaryCodeGen>("ATAN"));
  map->emplace("atanh", std::make_shared<TensorRTUnaryCodeGen>("ATANH"));
  map->emplace("ceil", std::make_shared<TensorRTUnaryCodeGen>("CEIL"));
  map->emplace("cos", std::make_shared<TensorRTUnaryCodeGen>("COS"));
  map->emplace("cosh", std::make_shared<TensorRTUnaryCodeGen>("COSH"));
  map->emplace("erf", std::make_shared<TensorRTUnaryCodeGen>("ERF"));
  map->emplace("exp", std::make_shared<TensorRTUnaryCodeGen>("EXP"));
  map->emplace("floor", std::make_shared<TensorRTUnaryCodeGen>("FLOOR"));
  map->emplace("log", std::make_shared<TensorRTUnaryCodeGen>("LOG"));
  map->emplace("negative", std::make_shared<TensorRTUnaryCodeGen>("NEG"));
  map->emplace("round", std::make_shared<TensorRTUnaryCodeGen>("ROUND"));
  map->emplace("sin", std::make_shared<TensorRTUnaryCodeGen>("SIN"));
  map->emplace("sinh", std::make_shared<TensorRTUnaryCodeGen>("SINH"));
  map->emplace("sqrt", std::make_shared<TensorRTUnaryCodeGen>("SQRT"));
  map->emplace("tan", std::make_shared<TensorRTUnaryCodeGen>("TAN"));

  // elemwise ops
  map->emplace("add", std::make_shared<TensorRTElemwiseCodeGen>("SUM"));
  map->emplace("divide", std::make_shared<TensorRTElemwiseCodeGen>("DIV"));
  map->emplace("equal", std::make_shared<TensorRTElemwiseCodeGen>("EQUAL"));
  map->emplace("floor_divide", std::make_shared<TensorRTElemwiseCodeGen>("FLOOR_DIV"));
  map->emplace("greater", std::make_shared<TensorRTElemwiseCodeGen>("GREATER"));
  map->emplace("less", std::make_shared<TensorRTElemwiseCodeGen>("LESS"));
  map->emplace("maximum", std::make_shared<TensorRTElemwiseCodeGen>("MAX"));
  map->emplace("minimum", std::make_shared<TensorRTElemwiseCodeGen>("MIN"));
  map->emplace("multiply", std::make_shared<TensorRTElemwiseCodeGen>("PROD"));
  map->emplace("power", std::make_shared<TensorRTElemwiseCodeGen>("POW"));
  map->emplace("subtract", std::make_shared<TensorRTElemwiseCodeGen>("SUB"));

  // reduce ops
  map->emplace("max", std::make_shared<TensorRTReduceCodeGen>("MAX"));
  map->emplace("mean", std::make_shared<TensorRTReduceCodeGen>("AVG"));
  map->emplace("min", std::make_shared<TensorRTReduceCodeGen>("MIN"));
  map->emplace("sum", std::make_shared<TensorRTReduceCodeGen>("SUM"));

  // math ops
  map->emplace("argmax", std::make_shared<TensorRTArgmaxminCodeGen>("MAX"));
  map->emplace("argmin", std::make_shared<TensorRTArgmaxminCodeGen>("MIN"));
  map->emplace("astype", std::make_shared<TensorRTAstypeCodeGen>("Identity"));
  map->emplace("concat", std::make_shared<TensorRTConcatCodeGen>("Concatenation"));
  map->emplace("expand_dims", std::make_shared<TensorRTReshapeCodeGen>("Shuffle"));
  map->emplace("matmul", std::make_shared<TensorRTMatmulCodeGen>("MatrixMultiply"));
  map->emplace("permute_dims", std::make_shared<TensorRTPermuteDimsCodeGen>("Shuffle"));
  map->emplace("reshape", std::make_shared<TensorRTReshapeCodeGen>("Shuffle"));
  map->emplace("square", std::make_shared<TensorRTSquareCodeGen>("ElementWise"));
  map->emplace("squeeze", std::make_shared<TensorRTReshapeCodeGen>("Shuffle"));
  map->emplace("strided_slice", std::make_shared<TensorRTStridedSliceCodeGen>("Slice"));
  map->emplace("take", std::make_shared<TensorRTTakeCodeGen>("Gather"));
  map->emplace("topk", std::make_shared<TensorRTTopkCodeGen>("TopK"));
  map->emplace("where", std::make_shared<TensorRTWhereCodeGen>("Select"));

  // create ops
  map->emplace("constant", std::make_shared<TensorRTConstantCodeGen>("Constant"));

  // activation ops
  map->emplace("clip", std::make_shared<TensorRTActivationCodeGen>("CLIP"));
  map->emplace("sigmoid", std::make_shared<TensorRTActivationCodeGen>("SIGMOID"));
  map->emplace("tanh", std::make_shared<TensorRTActivationCodeGen>("TANH"));
  map->emplace("nn.relu", std::make_shared<TensorRTActivationCodeGen>("RELU"));
  map->emplace("nn.leaky_relu", std::make_shared<TensorRTActivationCodeGen>("LEAKY_RELU"));

  // nn ops
  map->emplace("nn.adaptive_avg_pool2d",
               std::make_shared<TensorRTAdaptivePool2dCodeGen>("PoolingNd", "AVERAGE"));
  map->emplace("nn.avg_pool2d", std::make_shared<TensorRTPool2dCodeGen>("AVERAGE"));
  map->emplace("nn.batch_matmul", std::make_shared<TensorRTBatchMatmulCodeGen>("MatrixMultiply"));
  map->emplace("nn.conv2d", std::make_shared<TensorRTConvCodeGen>("ConvolutionNd", false));
  map->emplace("nn.max_pool2d", std::make_shared<TensorRTPool2dCodeGen>("MAX"));
  map->emplace("nn.pad", std::make_shared<TensorRTPadCodeGen>("Padding"));
  map->emplace("nn.softmax", std::make_shared<TensorRTSoftmaxCodeGen>("SoftMax"));

  // image ops
  map->emplace("image.resize2d", std::make_shared<TensorRTResize2dCodeGen>("Resize"));

  // special op
  map->emplace("input", std::make_shared<TensorRTInputCodeGen>("Input"));
  map->emplace("get_item", std::make_shared<TensorRTGetItemCodeGen>(""));
  map->emplace("tuple", std::make_shared<TensorRTTupleCodeGen>(""));
  map->emplace("plugin", std::make_shared<TensorRTPluginOpCodeGen>("PluginV2"));

  // msc ops
  map->emplace("msc.conv2d_bias", std::make_shared<TensorRTConvCodeGen>("ConvolutionNd", true));
  map->emplace("msc.linear", std::make_shared<TensorRTLinearCodeGen>("FullyConnected", false));
  map->emplace("msc.linear_bias", std::make_shared<TensorRTLinearCodeGen>("FullyConnected", true));

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
