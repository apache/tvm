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
 * \file src/contrib/msc/framework/torch/torch_opcode.cc
 */
#include "torch_opcode.h"

#include <memory>
#include <set>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> TorchOpCode::GetDocs() {
  stack_.Config(this);
  if (is_init()) {
    CodeGenInit();
  } else {
    CodeGenForward();
  }
  return stack_.GetDocs();
}

void TorchOpCode::CodeGenInit() {
  if (module_name().size() > 0) {
    stack_.op_start().op_end();
  } else {
    stack_.comment("passby: implement by " + func_name());
  }
}

void TorchOpCode::CodeGenForward() { stack_.op_start().op_inputs_arg(false).op_end(); }

const std::vector<int> TorchOpCode::GetPadding(const String& key) {
  std::vector<int> padding, src_padding;
  ICHECK(node()->GetAttr(key, &src_padding));
  if (node()->optype == "nn.conv1d" || node()->optype == "msc.conv1d_bias") {
    if (src_padding.size() == 2) {
      ICHECK(src_padding[0] == src_padding[1]) << "Only accept symmetric padding, get " << node();
      padding.push_back(src_padding[0]);
    } else {
      LOG_FATAL << "nn.conv1d with unexpected padding " << node();
    }
  } else if (node()->optype == "nn.conv2d" || node()->optype == "msc.conv2d_bias" ||
             node()->optype == "nn.avg_pool2d" || node()->optype == "nn.max_pool2d") {
    if (src_padding.size() == 4) {
      ICHECK(src_padding[0] == src_padding[2] && src_padding[1] == src_padding[3])
          << "Only accept symmetric padding, get " << node();
      padding.push_back(src_padding[0]);
      padding.push_back(src_padding[1]);
    } else {
      LOG_FATAL << "nn.conv2d/pool2d with unexpected padding " << node();
    }
  }
  return padding;
}

#define TORCH_OP_CODEGEN_METHODS(TypeName)                     \
 public:                                                       \
  TypeName(const String& module_name, const String& func_name) \
      : TorchOpCode(module_name, func_name) {}

class TorchAdaptivePoolCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchAdaptivePoolCodeGen);

 protected:
  void CodeGenInit() final { stack_.op_start().op_list_arg<int>("output_size").op_end(); }
};

class TorchAstypeCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchAstypeCodeGen);

 protected:
  void CodeGenForward() final {
    stack_.assign(IdxNode(), IdxInput())
        .inplace_start("to")
        .call_dtype_arg(node()->OutputAt(0)->dtype)
        .inplace_end();
  }
};

class TorchAttentionCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchAttentionCodeGen);

 protected:
  void CodeGenForward() final {
    std::string causal_mask;
    stack_.op_start().op_inputs_arg(false);
    if (node()->GetAttr("causal_mask", &causal_mask)) {
      if (causal_mask.size() > 0) {
        stack_.call_arg(true, "is_causal");
      }
    }
    stack_.op_end();
  }
};

class TorchAxesCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchAxesCodeGen);

 protected:
  void CodeGenInit() final {
    if (module_name().size() > 0) {
      const String& key = node()->HasAttr("axes") ? "axes" : "axis";
      stack_.op_start().op_list_arg<int>(key, "").op_end();
    } else {
      TorchOpCode::CodeGenInit();
    }
  }

  void CodeGenForward() final {
    if (module_name().size() > 0) {
      TorchOpCode::CodeGenForward();
    } else {
      const String& key = node()->HasAttr("axes") ? "axes" : "axis";
      stack_.op_start().op_input_arg().op_list_arg<int>(key, "").op_end();
    }
  }
};

class TorchAxisCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchAxisCodeGen);

 protected:
  void CodeGenInit() final {
    if (module_name().size() > 0) {
      stack_.op_start().op_arg<int>("axis", "dim").op_end();
    } else {
      TorchOpCode::CodeGenInit();
    }
  }

  void CodeGenForward() final {
    if (module_name().size() > 0) {
      TorchOpCode::CodeGenForward();
    } else {
      stack_.op_start().op_input_arg().op_arg<int>("axis", "dim").op_end();
    }
  }
};

class TorchBatchNormCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchBatchNormCodeGen);

 protected:
  void CodeGenInit() final {
    ICHECK(node()->GetTypeAttr<bool>("center") && node()->GetTypeAttr<bool>("scale"))
        << "Only support center and scale batchnorm, get " << node();
    const auto& gamma = node()->WeightAt("gamma");
    stack_.op_start()
        .call_arg(gamma->DimAt(0), "num_features")
        .op_arg<float>("epsilon", "eps")
        .op_end();
  }
};

class TorchBroadcastToCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchBroadcastToCodeGen);

 protected:
  void CodeGenForward() final {
    stack_.assign(IdxNode(), IdxInput())
        .inplace_start("expand")
        .call_list_arg(node()->GetTypeArrayAttr<int>("shape"), "", false, true)
        .inplace_end();
  }
};

class TorchClipCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchClipCodeGen);

 protected:
  void CodeGenForward() final {
    stack_.op_start().op_input_arg().op_arg<float>("min").op_arg<float>("max").op_end();
  }
};

class TorchConstantCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchConstantCodeGen);

 protected:
  void CodeGenInit() final {
    if (node()->HasAttr("scalar")) {
      if (node()->OutputAt(0)->DTypeName() == "int32") {
        stack_.assign(module_ref(), node()->GetTypeAttr<int>("scalar"));
      } else if (node()->OutputAt(0)->DTypeName() == "int64") {
        stack_.assign(module_ref(), node()->GetTypeAttr<int64_t>("scalar"));
      } else if (node()->OutputAt(0)->DTypeName() == "float32") {
        stack_.assign(module_ref(), node()->GetTypeAttr<float>("scalar"));
      }
    } else {
      stack_.call_start("torch.Tensor")
          .call_list_arg(node()->OutputAt(0)->shape, "", false, false)
          .call_end("data")
          .op_start()
          .call_arg("data")
          .op_end();
    }
  }

  void CodeGenForward() final { stack_.assign(IdxNode(), module_ref()); }
};

class TorchConvCodeGen : public TorchOpCode {
 public:
  TorchConvCodeGen(const String& module_name, const String& func_name, bool use_bias)
      : TorchOpCode(module_name, func_name), use_bias_(use_bias) {}

 protected:
  void CodeGenInit() final {
    const auto& weight = node()->WeightAt("weight");
    std::vector<int64_t> kernel_size;
    for (size_t i = 0; i < weight->Ndim(); i++) {
      if (weight->layout[i].name() == "I" || weight->layout[i].name() == "O") {
        continue;
      }
      kernel_size.push_back(weight->DimAt(i)->value);
    }
    stack_.op_start()
        .call_arg(weight->DimAt("I"), "in_channels")
        .call_arg(weight->DimAt("O"), "out_channels")
        .call_list_arg(kernel_size, "kernel_size")
        .op_list_arg<int>("strides", "stride")
        .call_list_arg(GetPadding(), "padding")
        .op_list_arg<int>("dilation")
        .op_arg<int>("groups")
        .call_arg(use_bias_, "bias")
        .op_end();
  }

 private:
  bool use_bias_;
};

class TorchCumsumCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchCumsumCodeGen);

 protected:
  void CodeGenForward() final {
    stack_.op_start()
        .op_input_arg()
        .op_arg<int>("axis", "dim")
        .call_dtype_arg(node()->OutputAt(0)->dtype, "dtype")
        .op_end();
  }
};

class TorchEmbeddingCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchEmbeddingCodeGen);

 protected:
  void CodeGenInit() final {
    const auto& weight = node()->WeightAt("weight");
    stack_.op_start()
        .call_arg(weight->DimAt("W"), "num_embeddings")
        .call_arg(weight->DimAt("E"), "embedding_dim")
        .op_end();
  }
};

class TorchExpandDimsCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchExpandDimsCodeGen);

 protected:
  void CodeGenForward() final {
    const auto& axes = node()->GetTypeArrayAttr<int>("axis");
    String idx_input = IdxInput();
    for (size_t i = 0; i < axes.size(); i++) {
      String idx_out = IdxNode();
      if (i < axes.size() - 1) {
        idx_out = idx_out + "_" + std::to_string(i);
      }
      stack_.op_start().call_arg(idx_input).call_arg(axes[i], "dim").op_end();
      idx_input = idx_out;
    }
  }
};

class TorchFullCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchFullCodeGen);

 protected:
  void CodeGenForward() final {
    stack_.op_start()
        .op_list_arg<int>("shape", "size")
        .op_input_arg(0, "fill_value")
        .call_dtype_arg(node()->OutputAt(0)->dtype, "dtype")
        .op_end();
  }
};

class TorchGetItemCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchGetItemCodeGen);

 protected:
  void CodeGenForward() final {
    stack_.assign(IdxNode(), IdxInput(node()->GetTypeAttr<int>("index")));
  }
};

class TorchGroupNormCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchGroupNormCodeGen);

 protected:
  void CodeGenInit() final {
    ICHECK(node()->GetTypeAttr<bool>("center") && node()->GetTypeAttr<bool>("scale"))
        << "Only support center and scale batchnorm, get " << node();
    int channel_axis = node()->GetTypeAttr<int>("channel_axis");
    stack_.op_start()
        .op_arg<int>("num_groups")
        .call_arg(node()->InputAt(0)->DimAt(channel_axis), "num_channels")
        .op_arg<float>("epsilon", "eps")
        .op_end();
  }
};

class TorchLayerNormCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchLayerNormCodeGen);

 protected:
  void CodeGenInit() final {
    ICHECK(node()->GetTypeAttr<bool>("center") && node()->GetTypeAttr<bool>("scale"))
        << "Only support center and scale batchnorm, get " << node();
    const auto& axes =
        CommonUtils::GetIndices(node()->GetTypeArrayAttr<int>("axes"), node()->InputAt(0)->Ndim());
    Array<Integer> normalized_shape;
    for (const auto& a : axes) {
      normalized_shape.push_back(node()->InputAt(0)->DimAt(a));
    }
    stack_.op_start()
        .call_list_arg(normalized_shape, "normalized_shape")
        .op_arg<float>("epsilon", "eps")
        .op_end();
  }
};

class TorchLinearCodeGen : public TorchOpCode {
 public:
  TorchLinearCodeGen(const String& module_name, const String& func_name, bool use_bias)
      : TorchOpCode(module_name, func_name), use_bias_(use_bias) {}

 protected:
  void CodeGenInit() final {
    const auto& weight = node()->WeightAt("weight");
    stack_.op_start()
        .call_arg(weight->DimAt("I"), "in_features")
        .call_arg(weight->DimAt("O"), "out_features")
        .call_arg(use_bias_, "bias")
        .op_end();
  }

 private:
  bool use_bias_;
};

class TorchNllLossCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchNllLossCodeGen);

 protected:
  void CodeGenForward() final {
    stack_.op_start()
        .op_inputs_arg(false)
        .op_str_arg("reduction")
        .op_arg<int>("ignore_index")
        .op_end();
  }
};

class TorchPoolCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchPoolCodeGen);

 protected:
  void CodeGenInit() final {
    stack_.op_start()
        .op_list_arg<int>("pool_size", "kernel_size")
        .op_list_arg<int>("strides", "stride")
        .call_list_arg(GetPadding(), "padding")
        .op_arg<bool>("ceil_mode");
    if (node()->optype == "nn.max_pool2d") {
      stack_.op_list_arg<int>("dilation");
    }
    stack_.op_end();
  }
};

class TorchReduceAxisCodeGen : public TorchOpCode {
 public:
  TorchReduceAxisCodeGen(const String& module_name, const String& func_name, bool as_list)
      : TorchOpCode(module_name, func_name), as_list_(as_list) {}

 protected:
  void CodeGenForward() final {
    stack_.op_start().op_input_arg();
    if (as_list_) {
      stack_.op_list_arg<int>("axis", "dim");
    } else {
      stack_.op_arg<int>("axis", "dim");
    }
    stack_.op_arg<bool>("keepdims", "keepdim").op_end();
  }

 private:
  bool as_list_;
};

class TorchRepeatCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchRepeatCodeGen);

 protected:
  void CodeGenForward() final {
    int repeat = node()->GetTypeAttr<int>("repeats");
    int axis = node()->GetTypeAttr<int>("axis");
    std::vector<int> repeats;
    for (size_t i = 0; i < node()->InputAt(0)->Ndim(); i++) {
      if (i == static_cast<size_t>(axis)) {
        repeats.push_back(repeat);
      } else {
        repeats.push_back(1);
      }
    }
    stack_.assign(IdxNode(), IdxInput())
        .inplace_start("repeat")
        .call_list_arg(repeats, "", false, true)
        .inplace_end();
  }
};

class TorchReshapeCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchReshapeCodeGen);

 protected:
  void CodeGenForward() final {
    std::vector<int> shape = node()->GetTypeArrayAttr<int>("shape");
    const auto& out_layout = node()->OutputAt(0)->layout;
    if (out_layout.defined()) {
      int32_t batch_dim = out_layout.IndexOf(tvm::tir::LayoutAxis::Get("N"));
      if (batch_dim > 0) {
        shape[batch_dim] = -1;
      }
    }
    stack_.op_start().op_input_arg().call_list_arg(shape).op_end();
  }
};

class TorchResize2dCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchResize2dCodeGen);

 protected:
  void CodeGenForward() final {
    const auto& method = node()->GetTypeAttr<std::string>("method");
    String v_method;
    if (method == "nearest_neighbor") {
      v_method = "nearest";
    } else {
      LOG(FATAL) << "Unexpected resize2d method " << method;
    }
    stack_.op_start()
        .op_input_arg()
        .op_list_arg<int>("size")
        .call_str_arg(v_method, "mode")
        .op_end();
  }
};

class TorchShapeCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchShapeCodeGen);

 protected:
  void CodeGenForward() final {
    if (node()->inputs.size() == 0) {
      stack_.op_start().op_list_arg<int>("shape", "").op_end();
    } else {
      stack_.assign(IdxNode(), IdxInput()).inplace_start("size").inplace_end();
    }
  }
};

class TorchSimpleCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchSimpleCodeGen);
};

class TorchSplitCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchSplitCodeGen)

 protected:
  void CodeGenForward() final {
    stack_.op_start().op_input_arg();
    std::vector<int64_t> indices;
    int axis = node()->GetTypeAttr<int>("axis");
    for (size_t i = 0; i < node()->outputs.size(); i++) {
      indices.push_back(node()->OutputAt(i)->DimAt(axis)->value);
    }
    stack_.call_list_arg(indices, "split_size_or_sections").op_arg<int>("axis", "dim").op_end();
  }
};

class TorchStridedSliceCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchStridedSliceCodeGen);

 protected:
  void CodeGenForward() final {
    const auto& begin = node()->GetTypeArrayAttr<int>("begin");
    const auto& end = node()->GetTypeArrayAttr<int>("end");
    const auto& strides = node()->GetTypeArrayAttr<int>("strides");
    const auto& axes =
        CommonUtils::GetIndices(node()->GetTypeArrayAttr<int>("axes"), node()->InputAt(0)->Ndim());
    std::set<size_t> axes_set;
    for (const auto& a : axes) {
      axes_set.insert(a);
    }
    Array<String> slice;
    for (size_t i = 0; i < node()->InputAt(0)->Ndim(); i++) {
      if (axes_set.count(i)) {
        slice.push_back(std::to_string(begin[i]) + ":" + std::to_string(end[i]) + ":" +
                        std::to_string(strides[i]));
      } else {
        slice.push_back(":");
      }
    }
    stack_.assign_index(IdxNode(), IdxInput(), slice);
  }
};

class TorchTriCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchTriCodeGen)

 protected:
  void CodeGenForward() final {
    stack_.op_start().op_input_arg().op_arg<int>("k", "diagonal").op_end();
  }
};

class TorchTupleCodeGen : public TorchOpCode {
  TORCH_OP_CODEGEN_METHODS(TorchTupleCodeGen)

 protected:
  void CodeGenForward() final { stack_.op_start().op_inputs_arg().op_end(); }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TorchOpCode>>> GetTorchOpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<TorchOpCode>>>();
  if (!map->empty()) return map;

  // simple ops
  map->emplace("abs", std::make_shared<TorchSimpleCodeGen>("", "torch.abs"));
  map->emplace("acos", std::make_shared<TorchSimpleCodeGen>("", "torch.acos"));
  map->emplace("acosh", std::make_shared<TorchSimpleCodeGen>("", "torch.acosh"));
  map->emplace("add", std::make_shared<TorchSimpleCodeGen>("", "torch.add"));
  map->emplace("asin", std::make_shared<TorchSimpleCodeGen>("", "torch.asin"));
  map->emplace("asinh", std::make_shared<TorchSimpleCodeGen>("", "torch.asinh"));
  map->emplace("atan", std::make_shared<TorchSimpleCodeGen>("", "torch.atan"));
  map->emplace("atanh", std::make_shared<TorchSimpleCodeGen>("", "torch.atanh"));
  map->emplace("bitwise_and", std::make_shared<TorchSimpleCodeGen>("", "torch.bitwise_and"));
  map->emplace("bitwise_not", std::make_shared<TorchSimpleCodeGen>("", "torch.bitwise_not"));
  map->emplace("bitwise_or", std::make_shared<TorchSimpleCodeGen>("", "torch.bitwise_or"));
  map->emplace("bitwise_xor", std::make_shared<TorchSimpleCodeGen>("", "torch.bitwise_xor"));
  map->emplace("ceil", std::make_shared<TorchSimpleCodeGen>("", "torch.ceil"));
  map->emplace("cos", std::make_shared<TorchSimpleCodeGen>("", "torch.cos"));
  map->emplace("cosh", std::make_shared<TorchSimpleCodeGen>("", "torch.cosh"));
  map->emplace("divide", std::make_shared<TorchSimpleCodeGen>("", "torch.divide"));
  map->emplace("exp", std::make_shared<TorchSimpleCodeGen>("", "torch.exp"));
  map->emplace("equal", std::make_shared<TorchSimpleCodeGen>("", "torch.equal"));
  map->emplace("floor", std::make_shared<TorchSimpleCodeGen>("", "torch.floor"));
  map->emplace("floor_divide", std::make_shared<TorchSimpleCodeGen>("", "torch.floor_divide"));
  map->emplace("greater", std::make_shared<TorchSimpleCodeGen>("", "torch.greater"));
  map->emplace("greater_equal", std::make_shared<TorchSimpleCodeGen>("", "torch.greater_equal"));
  map->emplace("less", std::make_shared<TorchSimpleCodeGen>("", "torch.less"));
  map->emplace("less_equal", std::make_shared<TorchSimpleCodeGen>("", "torch.less_equal"));
  map->emplace("log", std::make_shared<TorchSimpleCodeGen>("", "torch.log"));
  map->emplace("logical_and", std::make_shared<TorchSimpleCodeGen>("", "torch.logical_and"));
  map->emplace("logical_or", std::make_shared<TorchSimpleCodeGen>("", "torch.logical_or"));
  map->emplace("logical_xor", std::make_shared<TorchSimpleCodeGen>("", "torch.logical_xor"));
  map->emplace("matmul", std::make_shared<TorchSimpleCodeGen>("", "torch.matmul"));
  map->emplace("maximum", std::make_shared<TorchSimpleCodeGen>("", "torch.maximum"));
  map->emplace("minimum", std::make_shared<TorchSimpleCodeGen>("", "torch.minimum"));
  map->emplace("multiply", std::make_shared<TorchSimpleCodeGen>("", "torch.multiply"));
  map->emplace("negative", std::make_shared<TorchSimpleCodeGen>("", "torch.negative"));
  map->emplace("not_equal", std::make_shared<TorchSimpleCodeGen>("", "torch.not_equal"));
  map->emplace("power", std::make_shared<TorchSimpleCodeGen>("", "torch.pow"));
  map->emplace("round", std::make_shared<TorchSimpleCodeGen>("", "torch.round"));
  map->emplace("rsqrt", std::make_shared<TorchSimpleCodeGen>("", "torch.rsqrt"));
  map->emplace("sigmoid", std::make_shared<TorchSimpleCodeGen>("", "torch.sigmoid"));
  map->emplace("sign", std::make_shared<TorchSimpleCodeGen>("", "torch.sign"));
  map->emplace("sin", std::make_shared<TorchSimpleCodeGen>("", "torch.sin"));
  map->emplace("sinh", std::make_shared<TorchSimpleCodeGen>("", "torch.sinh"));
  map->emplace("square", std::make_shared<TorchSimpleCodeGen>("", "torch.square"));
  map->emplace("sqrt", std::make_shared<TorchSimpleCodeGen>("", "torch.sqrt"));
  map->emplace("subtract", std::make_shared<TorchSimpleCodeGen>("", "torch.subtract"));
  map->emplace("tan", std::make_shared<TorchSimpleCodeGen>("", "torch.tan"));
  map->emplace("tanh", std::make_shared<TorchSimpleCodeGen>("", "torch.tanh"));

  // reduce axis ops
  map->emplace("argmax", std::make_shared<TorchReduceAxisCodeGen>("", "torch.argmax", false));
  map->emplace("argmin", std::make_shared<TorchReduceAxisCodeGen>("", "torch.argmin", false));
  map->emplace("max", std::make_shared<TorchReduceAxisCodeGen>("", "torch.max", false));
  map->emplace("min", std::make_shared<TorchReduceAxisCodeGen>("", "torch.min", false));
  map->emplace("mean", std::make_shared<TorchReduceAxisCodeGen>("", "torch.mean", true));
  map->emplace("sum", std::make_shared<TorchReduceAxisCodeGen>("", "torch.sum", true));
  map->emplace("prod", std::make_shared<TorchReduceAxisCodeGen>("", "torch.prod", false));
  map->emplace("std", std::make_shared<TorchReduceAxisCodeGen>("", "torch.std", true));

  // axis && axes ops
  map->emplace("nn.log_softmax",
               std::make_shared<TorchAxisCodeGen>("nn.LogSoftmax", "functional.log_softmax"));
  map->emplace("nn.softmax",
               std::make_shared<TorchAxisCodeGen>("nn.Softmax", "functional.softmax"));
  map->emplace("permute_dims", std::make_shared<TorchAxesCodeGen>("", "torch.permute"));
  map->emplace("squeeze", std::make_shared<TorchAxesCodeGen>("", "torch.squeeze"));

  // math ops
  map->emplace("astype", std::make_shared<TorchAstypeCodeGen>("", "to"));
  map->emplace("broadcast_to", std::make_shared<TorchBroadcastToCodeGen>("", "expand"));
  map->emplace("clip", std::make_shared<TorchClipCodeGen>("", "torch.clamp"));
  map->emplace("cumsum", std::make_shared<TorchCumsumCodeGen>("", "torch.cumsum"));
  map->emplace("expand_dims", std::make_shared<TorchExpandDimsCodeGen>("", "torch.unsqueeze"));
  map->emplace("repeat", std::make_shared<TorchRepeatCodeGen>("", "repeat"));
  map->emplace("reshape", std::make_shared<TorchReshapeCodeGen>("", "torch.reshape"));
  map->emplace("split", std::make_shared<TorchSplitCodeGen>("", "torch.split"));
  map->emplace("strided_slice", std::make_shared<TorchStridedSliceCodeGen>("", ""));

  // create ops
  map->emplace("constant", std::make_shared<TorchConstantCodeGen>("nn.Parameter", ""));
  map->emplace("full", std::make_shared<TorchFullCodeGen>("", "torch.full"));
  map->emplace("tril", std::make_shared<TorchTriCodeGen>("", "torch.tril"));
  map->emplace("triu", std::make_shared<TorchTriCodeGen>("", "torch.triu"));

  // nn ops
  map->emplace("nn.adaptive_avg_pool2d",
               std::make_shared<TorchAdaptivePoolCodeGen>("nn.AdaptiveAvgPool2d",
                                                          "functional.adaptive_avg_pool2d"));
  map->emplace("nn.avg_pool2d",
               std::make_shared<TorchPoolCodeGen>("nn.AvgPool2d", "functional.avg_pool2d"));
  map->emplace("nn.batch_norm",
               std::make_shared<TorchBatchNormCodeGen>("nn.BatchNorm2d", "functional.batch_norm"));
  map->emplace("nn.conv1d",
               std::make_shared<TorchConvCodeGen>("nn.Conv1d", "functional.conv1d", false));
  map->emplace("nn.conv2d",
               std::make_shared<TorchConvCodeGen>("nn.Conv2d", "functional.conv2d", false));
  map->emplace("nn.gelu", std::make_shared<TorchSimpleCodeGen>("nn.GELU", "functional.gelu"));
  map->emplace("nn.group_norm",
               std::make_shared<TorchGroupNormCodeGen>("nn.GroupNorm", "functional.group_norm"));
  map->emplace("nn.layer_norm",
               std::make_shared<TorchLayerNormCodeGen>("nn.LayerNorm", "functional.layer_norm"));
  map->emplace("nn.linear",
               std::make_shared<TorchLinearCodeGen>("nn.Linear", "functional.linear", false));
  map->emplace("nn.max_pool2d",
               std::make_shared<TorchPoolCodeGen>("nn.MaxPool2d", "functional.max_pool2d"));
  map->emplace("nn.nll_loss", std::make_shared<TorchNllLossCodeGen>("", "functional.nll_loss"));
  map->emplace("nn.relu", std::make_shared<TorchSimpleCodeGen>("nn.ReLU", "functional.relu"));
  map->emplace("nn.silu", std::make_shared<TorchSimpleCodeGen>("nn.SiLU", "functional.silu"));

  // image ops
  map->emplace("image.resize2d",
               std::make_shared<TorchResize2dCodeGen>("", "torch.nn.functional.interpolate"));

  // special op
  map->emplace("get_item", std::make_shared<TorchGetItemCodeGen>("", ""));
  map->emplace("shape", std::make_shared<TorchShapeCodeGen>("", "torch.Size"));
  map->emplace("tuple", std::make_shared<TorchTupleCodeGen>("", "tuple"));

  // msc ops
  map->emplace("msc.attention", std::make_shared<TorchAttentionCodeGen>(
                                    "", "functional.scaled_dot_product_attention"));
  map->emplace("msc.conv1d_bias",
               std::make_shared<TorchConvCodeGen>("nn.Conv1d", "functional.conv1d", true));
  map->emplace("msc.conv2d_bias",
               std::make_shared<TorchConvCodeGen>("nn.Conv2d", "functional.conv2d", true));
  map->emplace("msc.embedding",
               std::make_shared<TorchEmbeddingCodeGen>("nn.Embedding", "functional.embedding"));
  map->emplace("msc.gelu", std::make_shared<TorchSimpleCodeGen>("nn.GELU", "functional.gelu"));
  map->emplace("msc.linear",
               std::make_shared<TorchLinearCodeGen>("nn.Linear", "functional.linear", false));
  map->emplace("msc.linear_bias",
               std::make_shared<TorchLinearCodeGen>("nn.Linear", "functional.linear", true));

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
