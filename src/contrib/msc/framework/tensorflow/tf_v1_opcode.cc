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
 * \file src/contrib/msc/framework/tensorflow/tf_v1_opcode.cc
 */
#include "tf_v1_opcode.h"

#include <memory>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> TFV1OpCode::GetDocs() {
  stack_.Config(this);
  CodeGenBuild();
  return stack_.GetDocs();
}

const std::pair<String, Array<String>> TFV1OpCode::GetPadding(const String& strides_key,
                                                              const String& kernel_key,
                                                              const String& padding_key) {
  String pad_mod = "";
  Array<String> padding;
  std::vector<int64_t> kernel_size;
  if (node()->optype == "nn.conv2d" || node()->optype == "msc.conv2d_bias") {
    const auto& weight = node()->WeightAt("weight");
    kernel_size.push_back(weight->DimAt("H")->value);
    kernel_size.push_back(weight->DimAt("W")->value);
  } else if (node()->optype == "nn.avg_pool2d" || node()->optype == "nn.max_pool2d") {
    ICHECK(node()->GetAttr(kernel_key, &kernel_size));
  } else {
    LOG_FATAL << "Unexpected padding node" << node();
  }
  const auto& strides = node()->GetTypeArrayAttr<int64_t>(strides_key);
  int64_t in_height = node()->InputAt(0)->DimAt("H")->value;
  int64_t in_width = node()->InputAt(0)->DimAt("W")->value;
  int64_t out_height = node()->OutputAt(0)->DimAt("H")->value;
  int64_t out_width = node()->OutputAt(0)->DimAt("W")->value;
  int64_t same_height = in_height / strides[0] + (in_height % strides[0] == 0 ? 0 : 1);
  int64_t same_width = in_width / strides[1] + (in_width % strides[1] == 0 ? 0 : 1);
  int64_t valid_height = (in_height - kernel_size[0] + 1) / strides[0];
  valid_height += (valid_height % strides[0] == 0 ? 0 : 1);
  int64_t valid_width = (in_width - kernel_size[1] + 1) / strides[1];
  valid_width += (valid_width % strides[1] == 0 ? 0 : 1);
  if (same_height == out_height && same_width == out_width) {
    pad_mod = "SAME";
  } else if (valid_height == out_height && valid_width == out_width) {
    pad_mod = "VALID";
  } else {
    const auto& src_padding = node()->GetTypeArrayAttr<int64_t>(padding_key);
    if (node()->optype == "nn.conv2d" || node()->optype == "msc.conv2d_bias" ||
        node()->optype == "nn.avg_pool2d" || node()->optype == "nn.max_pool2d") {
      const auto& out_layout = node()->GetTypeAttr<std::string>("out_layout");
      if (out_layout == "NHWC") {
        padding.push_back("[0, 0]");
      } else if (out_layout == "NCHW") {
        padding.push_back("[0, 0]");
        padding.push_back("[0, 0]");
      } else {
        LOG_FATAL << "Unexpected layout for padding node" << node();
      }
      if (src_padding.size() == 4) {
        padding.push_back("[" + std::to_string(src_padding[0]) + ", " +
                          std::to_string(src_padding[2]) + "]");
        padding.push_back("[" + std::to_string(src_padding[1]) + ", " +
                          std::to_string(src_padding[3]) + "]");
      } else {
        LOG_FATAL << "nn.conv2d/pool2d with unexpected padding " << node();
      }
      if (out_layout == "NHWC") {
        padding.push_back("[0, 0]");
      }
    } else {
      LOG_FATAL << "Unexpected padding node" << node();
    }
  }
  return std::make_pair(pad_mod, padding);
}

#define TFV1_OP_CODEGEN_METHODS(TypeName) \
 public:                                  \
  TypeName(const String& func_name) : TFV1OpCode(func_name) {}

class TFV1ArgMaxMinCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ArgMaxMinCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .op_arg<int>("axis")
        .op_dtype_arg(node()->OutputAt(0)->dtype, "output_type")
        .op_name_arg();
  }
};

class TFV1AstypeCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1AstypeCodeGen)

 protected:
  void CodeGenBuild() final {
    if (node()->InputAt(0)->dtype == node()->OutputAt(0)->dtype) {
      stack_.op_call("tf_v1.identity").op_input_arg().op_name_arg();
    } else {
      stack_.op_call().op_input_arg().op_dtype_arg(node()->OutputAt(0)->dtype).op_name_arg();
    }
  }
};

class TFV1AxesCodeGen : public TFV1OpCode {
 public:
  TFV1AxesCodeGen(const String& func_name, const String& attr_name) : TFV1OpCode(func_name) {
    attr_name_ = attr_name;
  }

 protected:
  void CodeGenBuild() final {
    const String& key = node()->HasAttr("axes") ? "axes" : "axis";
    stack_.op_call().op_input_arg().op_list_arg<int>(key, attr_name_).op_name_arg();
  }

 private:
  String attr_name_;
};

class TFV1AxisCodeGen : public TFV1OpCode {
 public:
  TFV1AxisCodeGen(const String& func_name, const String& attr_name) : TFV1OpCode(func_name) {
    attr_name_ = attr_name;
  }

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_arg<int>("axis", attr_name_).op_name_arg();
  }

 private:
  String attr_name_;
};

class TFV1BatchnormCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1BatchnormCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .op_arg<bool>("scale")
        .op_arg<bool>("center")
        .op_arg<float>("momentum")
        .op_arg<float>("epsilon");
    Array<String> weight_names{"gamma", "beta", "mean", "var"};
    Array<String> init_names{"gamma", "beta", "moving_mean", "moving_variance"};
    for (size_t i = 0; i < weight_names.size(); i++) {
      const auto& w_doc = DocUtils::ToStr(node()->WeightAt(weight_names[i])->name);
      stack_.inplace_start("tf_v1.constant_initializer", init_names[i] + "_initializer")
          .inplace_start("asnumpy", NullOpt, DocUtils::ToIndex("weights", w_doc))
          .inplace_end()
          .inplace_end();
    }
    stack_.op_name_arg();
  }
};

class TFV1BroadcastToCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1BroadcastToCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_list_arg<int>("shape").op_name_arg();
  }
};

class TFV1ClipCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ClipCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .op_arg<float>("min", "clip_value_min")
        .op_arg<float>("max", "clip_value_max")
        .op_name_arg();
  }
};

class TFV1ConcatCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ConcatCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg().op_arg<int>("axis").op_name_arg(); }
};

class TFV1ConstantCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ConstantCodeGen)

 protected:
  void CodeGenBuild() final { stack_.assign(IdxNode(), IdxWeight("const")); }
};

class TFV1ConvCodeGen : public TFV1OpCode {
 public:
  TFV1ConvCodeGen(const String& func_name, bool use_bias) : TFV1OpCode(func_name) {
    use_bias_ = use_bias;
  }

 protected:
  void CodeGenBuild() final {
    const auto& pair = GetPadding("strides");
    const auto& out_layout = node()->GetTypeAttr<std::string>("out_layout");
    int64_t groups = node()->GetTypeAttr<int64_t>("groups");
    std::vector<int> strides, dilation;
    const auto& attr_strides = node()->GetTypeArrayAttr<int>("strides");
    const auto& attr_dilation = node()->GetTypeArrayAttr<int>("dilation");
    if (out_layout == "NHWC") {
      strides = {1, attr_strides[0], attr_strides[1], 1};
      dilation = {1, attr_dilation[0], attr_dilation[1], 1};
    } else if (out_layout == "NCHW") {
      strides = {1, 1, attr_strides[0], attr_strides[1]};
      dilation = {1, 1, attr_dilation[0], attr_dilation[1]};
    } else {
      LOG_FATAL << "Unexpected layout for padding node" << node();
    }
    if (groups == 1) {
      stack_.op_call();
    } else if (groups == node()->InputAt(0)->DimAt("C")->value) {
      stack_.op_call("ops.nn_ops.depthwise_conv2d_native");
    } else {
      LOG_FATAL << "Unexpected conv with groups " << node();
    }
    stack_.op_input_arg()
        .op_weight_arg("weight")
        .call_arg(DocUtils::ToList(strides), "strides")
        .call_arg(DocUtils::ToList(dilation), "dilations")
        .op_str_arg("data_layout", "data_format");
    if (pair.first.size() > 0) {
      stack_.call_arg(DocUtils::ToStr(pair.first), "padding");
    } else if (pair.second.size() > 0) {
      stack_.call_arg(DocUtils::ToList(pair.second), "padding");
    } else {
      LOG_FATAL << "Can not parse padding for " << node();
    }
    stack_.op_name_arg();
    if (use_bias_) {
      stack_.op_call("ops.nn_ops.bias_add")
          .op_output_arg()
          .op_weight_arg("bias")
          .op_name_arg("name", node()->name + "_bias");
    }
  }

 private:
  bool use_bias_;
};

class TFV1CreateLikeCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1CreateLikeCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_input_arg().op_str_arg("dtype").op_name_arg(); }
};

class TFV1EinsumCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1EinsumCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& producer = node()->ProducerOf(0);
    stack_.op_call().op_str_arg("subscripts", "");
    if (node()->inputs.size() == 1 && producer->optype == "tuple") {
      stack_.call_arg(DocUtils::ToIndex(IdxInput(), 0));
    } else {
      stack_.op_inputs_arg(false);
    }
    stack_.op_name_arg();
  }
};

class TFV1FullCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1FullCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_list_arg<int>("shape", "").op_input_arg(0, "value").op_name_arg();
  }
};

class TFV1GetItemCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1GetItemCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.assign(IdxNode(), IdxInput(node()->GetTypeAttr<int>("index")));
  }
};

class TFV1PadCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1PadCodeGen)

 protected:
  void CodeGenBuild() final {
    String mode;
    const auto& attr_mode = node()->GetTypeAttr<std::string>("pad_mode");
    if (attr_mode == "constant") {
      mode = "CONSTANT";
    } else {
      LOG_FATAL << "Unexpected pad mode " << node();
    }
    Array<String> pad_width;
    const auto& attr_pad_width = node()->GetTypeArrayAttr<int>("pad_width");
    ICHECK(attr_pad_width.size() % 2 == 0) << "pad_width should be multiple of 2, get " << node();
    for (size_t i = 0; i < attr_pad_width.size(); i += 2) {
      const String& cur_pad = "[" + std::to_string(attr_pad_width[i]) + ", " +
                              std::to_string(attr_pad_width[i + 1]) + "]";
      pad_width.push_back(cur_pad);
    }
    const auto& val_producer = node()->ProducerOf(1);
    ICHECK(val_producer->optype == "constant" && val_producer->HasAttr("scalar"));
    stack_.op_call()
        .op_input_arg()
        .call_arg(DocUtils::ToList(pad_width), "paddings")
        .call_arg(DocUtils::ToStr(mode), "mode")
        .call_arg(val_producer->GetTypeAttr<float>("scalar"), "constant_values")
        .op_name_arg();
  }
};

class TFV1Pool2dCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1Pool2dCodeGen)

 protected:
  void CodeGenBuild() final {
    String pooling_type;
    if (node()->optype == "nn.avg_pool2d") {
      pooling_type = "AVG";
    } else if (node()->optype == "nn.max_pool2d") {
      pooling_type = "MAX";
    } else {
      LOG_FATAL << "Unexpected pool2d node " << node();
    }
    const auto& pair = GetPadding("strides", "pool_size");
    stack_.op_call()
        .op_input_arg()
        .op_list_arg<int>("pool_size", "window_shape")
        .call_arg(DocUtils::ToStr(pooling_type), "pooling_type")
        .op_list_arg<int>("dilation", "dilation_rate")
        .op_list_arg<int>("strides");
    if (pair.first.size() > 0) {
      stack_.call_arg(DocUtils::ToStr(pair.first), "padding");
    } else if (pair.second.size() > 0) {
      stack_.call_arg(DocUtils::ToList(pair.second), "padding");
    } else {
      LOG_FATAL << "Can not parse padding for " << node();
    }
    stack_.op_name_arg();
  }
};

class TFV1PermuteDimsCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1PermuteDimsCodeGen)

 protected:
  void CodeGenBuild() final {
    std::vector<int> axes;
    if (!node()->GetAttr("axes", &axes)) {
      for (size_t i = node()->InputAt(0)->Ndim(); i > 0; i--) {
        axes.push_back(i - 1);
      }
    }
    stack_.op_call().op_input_arg().call_arg(DocUtils::ToList(axes)).op_name_arg();
  }
};

class TFV1ReduceAxisCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ReduceAxisCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_list_arg<int>("axis").op_arg<bool>("keepdims").op_name_arg();
  }
};

class TFV1ReshapeCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1ReshapeCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_list_arg<int>("shape").op_name_arg();
  }
};

class TFV1Resize2dCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1Resize2dCodeGen)

 protected:
  void CodeGenBuild() final {
    String func_name;
    const auto& method = node()->GetTypeAttr<std::string>("method");
    const auto& coordinate_transformation_mode =
        node()->GetTypeAttr<std::string>("coordinate_transformation_mode");
    bool align_corners = coordinate_transformation_mode == "align_corners";
    if (method == "linear") {
      func_name = "tf_v1.image.resize_bilinear";
    } else if (method == "nearest_neighbor") {
      func_name = "tf_v1.image.resize_nearest_neighbor";
    } else {
      LOG_FATAL << "Unexpected resize with method " << node();
    }
    stack_.op_call(func_name)
        .op_input_arg()
        .op_list_arg<int>("size")
        .call_arg(align_corners, "align_corners")
        .op_name_arg();
  }
};

class TFV1SimpleCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1SimpleCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg(false).op_name_arg(); }
};

class TFV1SplitCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1SplitCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg();
    std::vector<int64_t> indices;
    int axis = node()->GetTypeAttr<int>("axis");
    for (size_t i = 0; i < node()->outputs.size(); i++) {
      indices.push_back(node()->OutputAt(i)->DimAt(axis)->value);
    }
    stack_.call_arg(DocUtils::ToList(indices), "num_or_size_splits")
        .op_arg<int>("axis")
        .op_name_arg();
  }
};

class TFV1StridedSliceCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1StridedSliceCodeGen)

 protected:
  void CodeGenBuild() final {
    std::vector<int> axes;
    if (!node()->GetAttr("axes", &axes)) {
      for (size_t i = 0; i < node()->InputAt(0)->Ndim(); i++) {
        axes.push_back(i);
      }
    }
    stack_.op_call()
        .op_input_arg()
        .op_list_arg<int>("begin")
        .op_list_arg<int>("end")
        .op_list_arg<int>("strides")
        .op_name_arg();
  }
};

class TFV1TakeCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1TakeCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_inputs_arg(false).op_arg<int>("axis").op_name_arg();
  }
};

class TFV1TileCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1TileCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_list_arg<int>("repeats", "multiples").op_name_arg();
  }
};

class TFV1TupleCodeGen : public TFV1OpCode {
  TFV1_OP_CODEGEN_METHODS(TFV1TupleCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg(); }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TFV1OpCode>>> GetTFV1OpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<TFV1OpCode>>>();
  if (!map->empty()) return map;
  // binary && unary ops
  map->emplace("abs", std::make_shared<TFV1SimpleCodeGen>("tf_v1.abs"));
  map->emplace("acos", std::make_shared<TFV1SimpleCodeGen>("tf_v1.acos"));
  map->emplace("acosh", std::make_shared<TFV1SimpleCodeGen>("tf_v1.acosh"));
  map->emplace("add", std::make_shared<TFV1SimpleCodeGen>("tf_v1.add"));
  map->emplace("asin", std::make_shared<TFV1SimpleCodeGen>("tf_v1.asin"));
  map->emplace("asinh", std::make_shared<TFV1SimpleCodeGen>("tf_v1.asinh"));
  map->emplace("atanh", std::make_shared<TFV1SimpleCodeGen>("tf_v1.atanh"));
  map->emplace("atan", std::make_shared<TFV1SimpleCodeGen>("tf_v1.atan"));
  map->emplace("ceil", std::make_shared<TFV1SimpleCodeGen>("tf_v1.ceil"));
  map->emplace("cos", std::make_shared<TFV1SimpleCodeGen>("tf_v1.cos"));
  map->emplace("cosh", std::make_shared<TFV1SimpleCodeGen>("tf_v1.cosh"));
  map->emplace("divide", std::make_shared<TFV1SimpleCodeGen>("tf_v1.divide"));
  map->emplace("equal", std::make_shared<TFV1SimpleCodeGen>("tf_v1.equal"));
  map->emplace("erf", std::make_shared<TFV1SimpleCodeGen>("tf_v1.erf"));
  map->emplace("exp", std::make_shared<TFV1SimpleCodeGen>("tf_v1.exp"));
  map->emplace("floor", std::make_shared<TFV1SimpleCodeGen>("tf_v1.floor"));
  map->emplace("floor_divide", std::make_shared<TFV1SimpleCodeGen>("tf_v1.floor_div"));
  map->emplace("floor_mod", std::make_shared<TFV1SimpleCodeGen>("tf_v1.floormod"));
  map->emplace("greater", std::make_shared<TFV1SimpleCodeGen>("tf_v1.greater"));
  map->emplace("greater_equal", std::make_shared<TFV1SimpleCodeGen>("tf_v1.greater_equal"));
  map->emplace("isfinite", std::make_shared<TFV1SimpleCodeGen>("tf_v1.is_finite"));
  map->emplace("isinf", std::make_shared<TFV1SimpleCodeGen>("tf_v1.is_inf"));
  map->emplace("isnan", std::make_shared<TFV1SimpleCodeGen>("tf_v1.is_nan"));
  map->emplace("less", std::make_shared<TFV1SimpleCodeGen>("tf_v1.less"));
  map->emplace("less_equal", std::make_shared<TFV1SimpleCodeGen>("tf_v1.less_equal"));
  map->emplace("log", std::make_shared<TFV1SimpleCodeGen>("tf_v1.log"));
  map->emplace("log1p", std::make_shared<TFV1SimpleCodeGen>("tf_v1.log1p"));
  map->emplace("logical_and", std::make_shared<TFV1SimpleCodeGen>("tf_v1.logical_and"));
  map->emplace("logical_or", std::make_shared<TFV1SimpleCodeGen>("tf_v1.logical_or"));
  map->emplace("logical_xor", std::make_shared<TFV1SimpleCodeGen>("tf_v1.logical_xor"));
  map->emplace("logical_not", std::make_shared<TFV1SimpleCodeGen>("tf_v1.logical_not"));
  map->emplace("maximum", std::make_shared<TFV1SimpleCodeGen>("tf_v1.maximum"));
  map->emplace("minimum", std::make_shared<TFV1SimpleCodeGen>("tf_v1.minimum"));
  map->emplace("multiply", std::make_shared<TFV1SimpleCodeGen>("tf_v1.multiply"));
  map->emplace("negative", std::make_shared<TFV1SimpleCodeGen>("tf_v1.negative"));
  map->emplace("not_equal", std::make_shared<TFV1SimpleCodeGen>("tf_v1.not_equal"));
  map->emplace("power", std::make_shared<TFV1SimpleCodeGen>("tf_v1.pow"));
  map->emplace("round", std::make_shared<TFV1SimpleCodeGen>("tf_v1.round"));
  map->emplace("rsqrt", std::make_shared<TFV1SimpleCodeGen>("tf_v1.rsqrt"));
  map->emplace("sigmoid", std::make_shared<TFV1SimpleCodeGen>("ops.math_ops.sigmoid"));
  map->emplace("sign", std::make_shared<TFV1SimpleCodeGen>("tf_v1.sign"));
  map->emplace("sin", std::make_shared<TFV1SimpleCodeGen>("tf_v1.sin"));
  map->emplace("sinh", std::make_shared<TFV1SimpleCodeGen>("tf_v1.sinh"));
  map->emplace("sqrt", std::make_shared<TFV1SimpleCodeGen>("tf_v1.sqrt"));
  map->emplace("subtract", std::make_shared<TFV1SimpleCodeGen>("tf_v1.subtract"));
  map->emplace("tan", std::make_shared<TFV1SimpleCodeGen>("tf_v1.tan"));
  map->emplace("tanh", std::make_shared<TFV1SimpleCodeGen>("tf_v1.tanh"));
  map->emplace("where", std::make_shared<TFV1SimpleCodeGen>("tf_v1.where"));

  // reduce axis ops
  map->emplace("max", std::make_shared<TFV1ReduceAxisCodeGen>("tf_v1.reduce_max"));
  map->emplace("min", std::make_shared<TFV1ReduceAxisCodeGen>("tf_v1.reduce_min"));
  map->emplace("mean", std::make_shared<TFV1ReduceAxisCodeGen>("tf_v1.reduce_mean"));
  map->emplace("sum", std::make_shared<TFV1ReduceAxisCodeGen>("tf_v1.reduce_sum"));
  map->emplace("prod", std::make_shared<TFV1ReduceAxisCodeGen>("tf_v1.reduce_prod"));
  map->emplace("std", std::make_shared<TFV1ReduceAxisCodeGen>("tf_v1.reduce_std"));

  // create ops
  map->emplace("constant", std::make_shared<TFV1ConstantCodeGen>("get_variable"));
  map->emplace("full", std::make_shared<TFV1FullCodeGen>("tf_v1.fill"));
  map->emplace("zeros_like", std::make_shared<TFV1CreateLikeCodeGen>("tf_v1.zeros_like"));

  // axis && axes ops
  map->emplace("expand_dims", std::make_shared<TFV1AxesCodeGen>("tf_v1.expand_dims", "axis"));
  map->emplace("nn.log_softmax", std::make_shared<TFV1AxisCodeGen>("tf_v1.nn.log_softmax", "axis"));
  map->emplace("nn.softmax", std::make_shared<TFV1AxisCodeGen>("tf_v1.nn.softmax", "axis"));
  map->emplace("squeeze", std::make_shared<TFV1AxesCodeGen>("ops.array_ops.squeeze", "axis"));

  // math ops
  map->emplace("argmax", std::make_shared<TFV1ArgMaxMinCodeGen>("tf_v1.argmax"));
  map->emplace("argmin", std::make_shared<TFV1ArgMaxMinCodeGen>("tf_v1.argmin"));
  map->emplace("astype", std::make_shared<TFV1AstypeCodeGen>("tf_v1.cast"));
  map->emplace("broadcast_to", std::make_shared<TFV1BroadcastToCodeGen>("tf_v1.broadcast_to"));
  map->emplace("clip", std::make_shared<TFV1ClipCodeGen>("tf_v1.clip_by_value"));
  map->emplace("concat", std::make_shared<TFV1ConcatCodeGen>("ops.array_ops.concat_v2"));
  map->emplace("concatenate", std::make_shared<TFV1ConcatCodeGen>("ops.array_ops.concat_v2"));
  map->emplace("einsum", std::make_shared<TFV1EinsumCodeGen>("tf_v1.einsum"));
  map->emplace("matmul", std::make_shared<TFV1SimpleCodeGen>("tf_v1.matmul"));
  map->emplace("permute_dims", std::make_shared<TFV1PermuteDimsCodeGen>("tf_v1.transpose"));
  map->emplace("reshape", std::make_shared<TFV1ReshapeCodeGen>("ops.array_ops.reshape"));
  map->emplace("split", std::make_shared<TFV1SplitCodeGen>("tf_v1.split"));
  map->emplace("strided_slice", std::make_shared<TFV1StridedSliceCodeGen>("tf_v1.strided_slice"));
  map->emplace("take", std::make_shared<TFV1TakeCodeGen>("tf_v1.gather"));
  map->emplace("tile", std::make_shared<TFV1TileCodeGen>("tf_v1.tile"));

  // nn ops
  map->emplace("nn.avg_pool2d", std::make_shared<TFV1Pool2dCodeGen>("ops.nn_ops.pool"));
  map->emplace("nn.batch_norm",
               std::make_shared<TFV1BatchnormCodeGen>("tf_v1.layers.batch_normalization"));
  map->emplace("nn.conv2d", std::make_shared<TFV1ConvCodeGen>("ops.nn_ops.conv2d", false));
  map->emplace("nn.max_pool2d", std::make_shared<TFV1Pool2dCodeGen>("ops.nn_ops.pool"));
  map->emplace("nn.pad", std::make_shared<TFV1PadCodeGen>("tf_v1.pad"));
  map->emplace("nn.relu", std::make_shared<TFV1SimpleCodeGen>("tf_v1.nn.relu"));

  // image ops
  map->emplace("image.resize2d",
               std::make_shared<TFV1Resize2dCodeGen>("tf_v1.image.resize_nearest_neighbor"));

  // special op
  map->emplace("get_item", std::make_shared<TFV1GetItemCodeGen>(""));
  map->emplace("tuple", std::make_shared<TFV1TupleCodeGen>("tuple"));

  // msc ops
  map->emplace("msc.conv2d", std::make_shared<TFV1ConvCodeGen>("ops.nn_ops.conv2d", false));
  map->emplace("msc.conv2d_bias", std::make_shared<TFV1ConvCodeGen>("ops.nn_ops.conv2d", true));

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
