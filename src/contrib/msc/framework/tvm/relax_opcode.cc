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
 * \file src/contrib/msc/framework/tvm/relax_opcode.cc
 */
#include "relax_opcode.h"

#include <memory>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> RelaxOpCode::GetDocs() {
  stack_.Config(this);
  CodeGenBuild();
  bool emit_var = true;
  if (node()->optype == "input" || node()->optype == "constant" || node()->optype == "shape") {
    emit_var = false;
  }
  if (emit_var) {
    const auto& name = config()->explicit_name ? node()->name : "";
    BuilderEmit(IdxNode(), name);
  }
  return stack_.GetDocs();
}

void RelaxOpCode::BuilderEmit(const String& ret, const String& name) {
  stack_.func_call("block_builder.emit", ret).call_arg(ret);
  if (name.size() > 0) {
    stack_.call_arg(DocUtils::ToStr(name), "name_hint");
  }
}

const ExprDoc RelaxOpCode::GetOutDtype(const String& key, int input_idx) {
  if (config()->use_tools && input_idx >= 0 &&
      node()->inputs.size() > static_cast<size_t>(input_idx)) {
    return DocUtils::ToDoc(IdxInput(input_idx) + ".struct_info.dtype");
  }
  std::string out_dtype;
  if (!node()->GetAttr(key, &out_dtype) && config()->from_relay) {
    return DocUtils::ToStr(node()->OutputAt(0)->DTypeName());
  }
  return DocUtils::ToStr(out_dtype);
}

const std::vector<int> RelaxOpCode::GetAxes(const String& key) {
  std::vector<int> axes;
  int axis;
  if (!node()->GetAttr(key, &axes) && node()->GetAttr(key, &axis)) {
    axes.push_back(axis);
  }
  return axes;
}

#define RELAX_OP_CODEGEN_METHODS(TypeName) \
 public:                                   \
  TypeName(const String& func_name) : RelaxOpCode(func_name) {}

class RelaxAdaptivePool2dCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxAdaptivePool2dCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .op_list_arg<int>("output_size")
        .op_str_arg("layout")
        .op_str_arg("out_layout");
  }
};

class RelaxAstypeCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxAstypeCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_input_arg().op_str_arg("dtype"); }
};

class RelaxAttentionCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxAttentionCodeGen)

 protected:
  void CodeGenBuild() final {
    for (size_t i = 0; i < 3; i++) {
      const String& axes_key = i == 0 ? "axes" : "axes_" + std::to_string(i);
      stack_.op_call("relax.op.permute_dims", IdxInput(i))
          .op_input_arg(i)
          .op_list_arg<int>(axes_key, "axes");
    }
    stack_.op_call().op_inputs_arg(false).op_arg<float>("scale").op_str_arg("causal_mask");
    stack_.op_call("relax.op.permute_dims").op_output_arg().op_list_arg<int>("axes_3", "axes");
  }
};

class RelaxAxisCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxAxisCodeGen)

 protected:
  void CodeGenBuild() final {
    std::vector<int> axes = GetAxes("axis");
    stack_.op_call().op_input_arg();
    if (axes.size() > 0) {
      stack_.call_arg(axes[0], "axis");
    }
  }
};

class RelaxAxesCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxAxesCodeGen)

 protected:
  void CodeGenBuild() final {
    const String& key = node()->HasAttr("axes") ? "axes" : "axis";
    stack_.op_call().op_input_arg().call_arg(DocUtils::ToList(GetAxes(key)), key);
  }
};

class RelaxBatchMatmulCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxBatchMatmulCodeGen)

 protected:
  void CodeGenBuild() final {
    bool transpose_a = node()->GetTypeAttr<bool>("transpose_a");
    bool transpose_b = node()->GetTypeAttr<bool>("transpose_b");
    if (!transpose_a && !transpose_b) {
      stack_.op_call().op_inputs_arg(false).op_str_arg("out_dtype");
    } else if (transpose_a && !transpose_b) {
      std::vector<size_t> axes;
      for (size_t i = 0; i < node()->InputAt(0)->Ndim() - 2; i++) {
        axes.push_back(i);
      }
      axes.push_back(node()->InputAt(0)->Ndim() - 1);
      axes.push_back(node()->InputAt(0)->Ndim() - 2);
      stack_.op_call("relax.op.permute_dims", IdxInput(0))
          .op_input_arg()
          .call_arg(DocUtils::ToList(axes));
      BuilderEmit(IdxInput(0));
      stack_.op_call().op_inputs_arg(false).op_str_arg("out_dtype");
    } else if (!transpose_a && transpose_b) {
      std::vector<size_t> axes;
      for (size_t i = 0; i < node()->InputAt(1)->Ndim() - 2; i++) {
        axes.push_back(i);
      }
      axes.push_back(node()->InputAt(1)->Ndim() - 1);
      axes.push_back(node()->InputAt(1)->Ndim() - 2);
      stack_.op_call("relax.op.permute_dims", IdxInput(1))
          .op_input_arg(1)
          .call_arg(DocUtils::ToList(axes));
      BuilderEmit(IdxInput(1));
      stack_.op_call().op_inputs_arg(false).op_str_arg("out_dtype");
    } else {
      for (size_t idx = 0; idx < 2; idx++) {
        std::vector<size_t> axes;
        for (size_t i = 0; i < node()->InputAt(idx)->Ndim() - 2; i++) {
          axes.push_back(i);
        }
        axes.push_back(node()->InputAt(idx)->Ndim() - 1);
        axes.push_back(node()->InputAt(idx)->Ndim() - 2);
        stack_.op_call("relax.op.permute_dims", IdxInput(idx))
            .op_input_arg(idx)
            .call_arg(DocUtils::ToList(axes));
        BuilderEmit(IdxInput(idx));
      }
      stack_.op_call().op_inputs_arg(false).op_str_arg("out_dtype");
    }
  }
};

class RelaxBatchNormCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxBatchNormCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .op_weight_arg("gamma")
        .op_weight_arg("beta")
        .op_weight_arg("mean")
        .op_weight_arg("var")
        .op_arg<int>("axis")
        .op_arg<float>("epsilon")
        .op_arg<bool>("center")
        .op_arg<bool>("scale")
        .op_arg<float>("momentum");
  }
};

class RelaxBiasAddCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxBiasAddCodeGen)

 protected:
  void CodeGenBuild() final {
    int axis = CommonUtils::GetIndex(node()->GetTypeAttr<int>("axis"), node()->OutputAt(0)->Ndim());
    Array<Integer> expand_shape;
    for (size_t i = 0; i < node()->InputAt(0)->Ndim(); i++) {
      if (i == static_cast<size_t>(axis)) {
        expand_shape.push_back(node()->InputAt(0)->DimAt(i));
      } else {
        expand_shape.push_back(Integer(1));
      }
    }
    stack_.op_call("relax.op.reshape", IdxInput(1))
        .op_input_arg(1)
        .call_arg(DocUtils::ToList(expand_shape), "shape");
    BuilderEmit(IdxInput(1));
    stack_.op_call().op_inputs_arg(false);
  }
};

class RelaxBroadcastToCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxBroadcastToCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_input_arg().op_list_arg<int>("shape"); }
};

class RelaxClipCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxClipCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg();
    if (config()->from_relay) {
      stack_.op_arg<float>("a_min", "min").op_arg<float>("a_max", "max");
    } else {
      stack_.op_arg<float>("min").op_arg<float>("max");
    }
  }
};

class RelaxConcatCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxConcatCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg().op_arg<int>("axis"); }
};

class RelaxConstantCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxConstantCodeGen)

 protected:
  void CodeGenBuild() final { stack_.assign(IdxNode(), IdxWeight("const")); }
};

class RelaxConvCodeGen : public RelaxOpCode {
 public:
  RelaxConvCodeGen(const String& func_name, bool use_bias)
      : RelaxOpCode(func_name), use_bias_(use_bias) {}

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .op_weight_arg("weight")
        .op_list_arg<int>("strides")
        .op_list_arg<int>("padding")
        .op_list_arg<int>("dilation")
        .op_arg<int>("groups")
        .op_str_arg("data_layout")
        .op_str_arg("kernel_layout")
        .op_str_arg("out_layout")
        .call_arg(GetOutDtype(), "out_dtype");
    if (use_bias_) {
      std::string out_layout_str;
      if (!node()->GetAttr("out_layout", &out_layout_str)) {
        ICHECK(node()->GetAttr("data_layout", &out_layout_str))
            << "out_layout or data_layout should be given, get " << node();
      }
      const auto& out_layout = tir::Layout(out_layout_str);
      Array<Integer> expand_shape;
      for (size_t i = 0; i < node()->OutputAt(0)->Ndim(); i++) {
        if (out_layout[i].name() == "C") {
          expand_shape.push_back(node()->OutputAt(0)->DimAt(i));
        } else {
          expand_shape.push_back(Integer(1));
        }
      }
      BuilderEmit(IdxNode());
      stack_.func_call("relax.op.reshape", "expand_bias")
          .op_weight_arg("bias")
          .call_arg(DocUtils::ToList(expand_shape), "shape");
      BuilderEmit("expand_bias");
      stack_.func_call("relax.op.add", IdxNode()).call_arg(IdxNode()).call_arg("expand_bias");
    }
  }

 private:
  bool use_bias_;
};

class RelaxCreateCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxCreateCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_list_arg<int>("shape").op_str_arg("dtype"); }
};

class RelaxCreateLikeCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxCreateLikeCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_input_arg().op_str_arg("dtype"); }
};

class RelaxCumsumCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxCumsumCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_arg<int>("axis").op_str_arg("dtype");
  }
};

class RelaxEinsumCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxEinsumCodeGen)

 protected:
  void CodeGenBuild() final {
    const String& key = config()->from_relay ? "equation" : "subscripts";
    stack_.op_call().op_inputs_arg().op_str_arg(key, "subscripts");
  }
};

class RelaxStridedSliceCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxStridedSliceCodeGen)

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
        .call_arg(DocUtils::ToList(axes), "axes")
        .op_list_arg<int>("begin")
        .op_list_arg<int>("end")
        .op_list_arg<int>("strides");
  }
};

class RelaxEmbeddingCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxEmbeddingCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& input = node()->InputAt(0);
    if (input->DTypeName() != "int32") {
      stack_.op_call("relax.op.astype", IdxInput())
          .op_input_arg()
          .call_arg(DocUtils::ToStr("int32"));
      BuilderEmit(IdxInput());
    }
    if (input->Ndim() > 1) {
      stack_.op_call("relax.op.reshape", IdxInput())
          .op_input_arg()
          .call_arg(DocUtils::ToList(std::vector<int>{-1}), "shape");
      BuilderEmit(IdxInput());
    }
    stack_.op_call().op_weight_arg("weight").op_input_arg().op_arg<int>("axis");
    if (input->Ndim() > 1) {
      BuilderEmit(IdxNode());
      stack_.op_call("relax.op.reshape")
          .op_output_arg()
          .call_arg(DocUtils::ToList(node()->OutputAt(0)->shape));
    }
  }
};

class RelaxFullCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxFullCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_list_arg<int>("shape").op_input_arg(0, "fill_value").op_str_arg("dtype");
  }
};

class RelaxGetItemCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxGetItemCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& producer = node()->ProducerOf(0);
    stack_.op_call("msc::auto", IdxNode()).call_arg(IdxNodeBase(producer)).op_arg<int>("index");
  }
};

class RelaxGroupNormCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxGroupNormCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_weight_arg("gamma").op_weight_arg("beta").op_arg<int>(
        "num_groups");
    if (config()->from_relay) {
      std::vector<size_t> axes;
      for (size_t i = 2; i < node()->InputAt(0)->Ndim(); i++) {
        axes.push_back(i);
      }
      stack_.op_arg<int>("axis", "channel_axis").call_arg(DocUtils::ToList(axes), "axes");
    } else {
      stack_.op_arg<int>("channel_axis").op_list_arg<int>("axes");
    }
    stack_.op_arg<float>("epsilon").op_arg<bool>("center").op_arg<bool>("scale");
  }
};

class RelaxLayerNormCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxLayerNormCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_weight_arg("gamma").op_weight_arg("beta");
    if (config()->from_relay) {
      stack_.op_arg<int>("axis", "axes");
    } else {
      stack_.op_list_arg<int>("axes");
    }
    stack_.op_arg<float>("epsilon").op_arg<bool>("center").op_arg<bool>("scale");
  }
};

class RelaxLinearCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxLinearCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call();
    if (node()->inputs.size() == 1) {
      stack_.op_input_arg().op_weight_arg("weight").op_weight_arg("bias");
    } else {
      stack_.op_inputs_arg(false);
    }
    stack_.call_arg(GetOutDtype(), "out_dtype");
  }
};

class RelaxMatmulCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxMatmulCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_inputs_arg(false).call_arg(GetOutDtype(), "out_dtype");
  }
};

class RelaxNllLossCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxNllLossCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_inputs_arg(false).op_str_arg("reduction").op_arg<int>("ignore_index");
  }
};

class RelaxPadCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxPadCodeGen)

 protected:
  void CodeGenBuild() final {
    Array<String> pad_width;
    const auto& attr_pad_width = node()->GetTypeArrayAttr<int>("pad_width");
    ICHECK(attr_pad_width.size() % 2 == 0) << "pad_width should be multiple of 2, get " << node();
    for (size_t i = 0; i < attr_pad_width.size(); i += 2) {
      const String& cur_pad = "[" + std::to_string(attr_pad_width[i]) + ", " +
                              std::to_string(attr_pad_width[i + 1]) + "]";
      pad_width.push_back(cur_pad);
    }
    stack_.op_call()
        .op_input_arg()
        .op_list_arg<int>("pad_width")
        .op_input_arg(1, "pad_value")
        .op_str_arg("pad_mode");
  }
};

class RelaxPool2dCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxPool2dCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call()
        .op_input_arg()
        .op_list_arg<int>("pool_size")
        .op_list_arg<int>("strides")
        .op_list_arg<int>("padding")
        .op_list_arg<int>("dilation")
        .op_arg<bool>("ceil_mode")
        .op_str_arg("layout")
        .op_str_arg("out_layout");
  }
};

class RelaxPermuteDimsCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxPermuteDimsCodeGen)

 protected:
  void CodeGenBuild() final {
    std::vector<int> axes;
    if (!node()->GetAttr("axes", &axes)) {
      for (size_t i = node()->InputAt(0)->Ndim(); i > 0; i--) {
        axes.push_back(i - 1);
      }
    }
    stack_.op_call().op_input_arg().call_arg(DocUtils::ToList(axes), "axes");
  }
};

class RelaxReduceAxisCodeGen : public RelaxOpCode {
 public:
  RelaxReduceAxisCodeGen(const String& func_name, bool as_list)
      : RelaxOpCode(func_name), as_list_(as_list) {}

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg();
    std::vector<int> axes = GetAxes("axis");
    if (as_list_) {
      stack_.call_arg(DocUtils::ToList(axes), "axis");
    } else if (axes.size() > 0) {
      stack_.call_arg(axes[0], "axis");
    }
    stack_.op_arg<bool>("keepdims");
  }

 private:
  bool as_list_;
};

class RelaxRepeatCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxRepeatCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg().op_arg<int>("repeats").op_arg<int>("axis");
  }
};

class RelaxReshapeCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxReshapeCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& out_shape = GetPrims(node()->OutputAt(0));
    stack_.op_call().op_input_arg().call_arg(DocUtils::ToList(out_shape), "shape");
  }
};

class RelaxScatterElementsCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxScatterElementsCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg(false).op_arg<int>("axis"); }
};

class RelaxScatterNDCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxScatterNDCodeGen)

 protected:
  void CodeGenBuild() final {
    if (config()->from_relay) {
      size_t ndim = node()->InputAt(1)->Ndim();
      std::vector<size_t> axes;
      axes.push_back(ndim - 1);
      for (size_t i = 0; i < ndim - 1; i++) {
        axes.push_back(i);
      }
      stack_.func_call("relax.op.permute_dims", IdxInput(1))
          .call_arg(IdxInput(1))
          .call_arg(DocUtils::ToList(axes));
      BuilderEmit(IdxInput(1), "permute_" + std::to_string(node()->index));
    }
    stack_.op_call().op_inputs_arg(false).op_str_arg("mode", "reduction");
  }
};

class RelaxResize2dCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxResize2dCodeGen)

 protected:
  void CodeGenBuild() final {
    // roi has forced to be float list
    Array<String> roi_list;
    std::vector<float> roi = node()->GetTypeArrayAttr<float>("roi");
    for (const auto& r : roi) {
      roi_list.push_back("float(" + std::to_string(r) + ")");
    }
    stack_.op_call()
        .op_input_arg()
        .func_call("relax.ShapeExpr")
        .op_list_arg<int>("size", "values")
        .pop_nest()
        .call_arg(DocUtils::ToList(roi_list))
        .op_str_arg("layout")
        .op_str_arg("method")
        .op_str_arg("coordinate_transformation_mode")
        .op_str_arg("rounding_method")
        .op_arg<float>("cubic_alpha")
        .op_arg<int>("cubic_exclude")
        .op_arg<float>("extrapolation_value")
        .op_str_arg("out_dtype");
  }
};

class RelaxShapeCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxShapeCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_list_arg<int>("shape", "values"); }
};

class RelaxSimpleCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxSimpleCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg(false); }
};

class RelaxSplitCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxSplitCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_input_arg();
    int sections;
    if (node()->GetAttr("indices_or_sections", &sections)) {
      stack_.op_arg<int>("indices_or_sections");
    } else {
      stack_.op_list_arg<int>("indices_or_sections");
    }
    stack_.op_arg<int>("axis");
  }
};

class RelaxStackCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxStackCodeGen)

 protected:
  void CodeGenBuild() final {
    stack_.op_call().op_inputs_arg().op_arg<int>("axis");
    BuilderEmit(IdxNode(), "cat_" + std::to_string(node()->index));
    const auto& out_shape = GetPrims(node()->OutputAt(0));
    stack_.func_call("relax.op.reshape", IdxNode())
        .call_arg(IdxNode())
        .call_arg(DocUtils::ToList(out_shape), "shape");
  }
};

class RelaxTakeCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxTakeCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg(false).op_arg<int>("axis"); }
};

class RelaxTileCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxTileCodeGen)

 protected:
  void CodeGenBuild() final {
    const String& key = config()->from_relay ? "reps" : "repeats";
    stack_.op_call().op_input_arg().op_list_arg<int>(key, "repeats");
  }
};

class RelaxTupleCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxTupleCodeGen)

 protected:
  void CodeGenBuild() final { stack_.op_call().op_inputs_arg(); }
};

class RelaxTriCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxTriCodeGen)

 protected:
  void CodeGenBuild() final {
    if (node()->optype == "trilu") {
      const String& func_name =
          node()->GetTypeAttr<bool>("upper") ? "relax.op.triu" : "relax.op.tril";
      stack_.op_call(func_name).op_input_arg().op_arg<int>("k");
    } else {
      stack_.op_call().op_input_arg().op_arg<int>("k");
    }
  }
};

class RelaxPluginOpCodeGen : public RelaxOpCode {
  RELAX_OP_CODEGEN_METHODS(RelaxPluginOpCodeGen)

 protected:
  void CodeGenBuild() final {
    const auto& plugin = GetPlugin(node()->optype);
    stack_.op_call("plugin." + node()->optype).op_inputs_arg(false);
    for (const auto& a : plugin->attrs) {
      stack_.call_arg(GetAttrDoc(a->name, a->type), a->name);
    }
  }
};

const std::shared_ptr<std::unordered_map<String, std::shared_ptr<RelaxOpCode>>> GetRelaxOpCodes() {
  static auto map = std::make_shared<std::unordered_map<String, std::shared_ptr<RelaxOpCode>>>();
  if (!map->empty()) return map;
  // binary && unary ops
  map->emplace("abs", std::make_shared<RelaxSimpleCodeGen>("relax.op.abs"));
  map->emplace("acos", std::make_shared<RelaxSimpleCodeGen>("relax.op.acos"));
  map->emplace("acosh", std::make_shared<RelaxSimpleCodeGen>("relax.op.acosh"));
  map->emplace("add", std::make_shared<RelaxSimpleCodeGen>("relax.op.add"));
  map->emplace("asin", std::make_shared<RelaxSimpleCodeGen>("relax.op.asin"));
  map->emplace("asinh", std::make_shared<RelaxSimpleCodeGen>("relax.op.asinh"));
  map->emplace("atan", std::make_shared<RelaxSimpleCodeGen>("relax.op.atan"));
  map->emplace("atanh", std::make_shared<RelaxSimpleCodeGen>("relax.op.atanh"));
  map->emplace("bitwise_and", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_and"));
  map->emplace("bitwise_not", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_not"));
  map->emplace("bitwise_or", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_or"));
  map->emplace("bitwise_xor", std::make_shared<RelaxSimpleCodeGen>("relax.op.bitwise_xor"));
  map->emplace("ceil", std::make_shared<RelaxSimpleCodeGen>("relax.op.ceil"));
  map->emplace("cos", std::make_shared<RelaxSimpleCodeGen>("relax.op.cos"));
  map->emplace("cosh", std::make_shared<RelaxSimpleCodeGen>("relax.op.cosh"));
  map->emplace("divide", std::make_shared<RelaxSimpleCodeGen>("relax.op.divide"));
  map->emplace("equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.equal"));
  map->emplace("erf", std::make_shared<RelaxSimpleCodeGen>("relax.op.erf"));
  map->emplace("exp", std::make_shared<RelaxSimpleCodeGen>("relax.op.exp"));
  map->emplace("floor", std::make_shared<RelaxSimpleCodeGen>("relax.op.floor"));
  map->emplace("floor_divide", std::make_shared<RelaxSimpleCodeGen>("relax.op.floor_divide"));
  map->emplace("greater", std::make_shared<RelaxSimpleCodeGen>("relax.op.greater"));
  map->emplace("greater_equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.greater_equal"));
  map->emplace("isfinite", std::make_shared<RelaxSimpleCodeGen>("relax.op.isfinite"));
  map->emplace("isinf", std::make_shared<RelaxSimpleCodeGen>("relax.op.isinf"));
  map->emplace("isnan", std::make_shared<RelaxSimpleCodeGen>("relax.op.isnan"));
  map->emplace("less", std::make_shared<RelaxSimpleCodeGen>("relax.op.less"));
  map->emplace("less_equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.less_equal"));
  map->emplace("log", std::make_shared<RelaxSimpleCodeGen>("relax.op.log"));
  map->emplace("logical_and", std::make_shared<RelaxSimpleCodeGen>("relax.op.logical_and"));
  map->emplace("logical_or", std::make_shared<RelaxSimpleCodeGen>("relax.op.logical_or"));
  map->emplace("logical_xor", std::make_shared<RelaxSimpleCodeGen>("relax.op.logical_xor"));
  map->emplace("logical_not", std::make_shared<RelaxSimpleCodeGen>("relax.op.logical_not"));
  map->emplace("maximum", std::make_shared<RelaxSimpleCodeGen>("relax.op.maximum"));
  map->emplace("minimum", std::make_shared<RelaxSimpleCodeGen>("relax.op.minimum"));
  map->emplace("multiply", std::make_shared<RelaxSimpleCodeGen>("relax.op.multiply"));
  map->emplace("negative", std::make_shared<RelaxSimpleCodeGen>("relax.op.negative"));
  map->emplace("not_equal", std::make_shared<RelaxSimpleCodeGen>("relax.op.not_equal"));
  map->emplace("power", std::make_shared<RelaxSimpleCodeGen>("relax.op.power"));
  map->emplace("round", std::make_shared<RelaxSimpleCodeGen>("relax.op.round"));
  map->emplace("rsqrt", std::make_shared<RelaxSimpleCodeGen>("relax.op.rsqrt"));
  map->emplace("sigmoid", std::make_shared<RelaxSimpleCodeGen>("relax.op.sigmoid"));
  map->emplace("sign", std::make_shared<RelaxSimpleCodeGen>("relax.op.sign"));
  map->emplace("sin", std::make_shared<RelaxSimpleCodeGen>("relax.op.sin"));
  map->emplace("sinh", std::make_shared<RelaxSimpleCodeGen>("relax.op.sinh"));
  map->emplace("square", std::make_shared<RelaxSimpleCodeGen>("relax.op.square"));
  map->emplace("sqrt", std::make_shared<RelaxSimpleCodeGen>("relax.op.sqrt"));
  map->emplace("subtract", std::make_shared<RelaxSimpleCodeGen>("relax.op.subtract"));
  map->emplace("tan", std::make_shared<RelaxSimpleCodeGen>("relax.op.tan"));
  map->emplace("tanh", std::make_shared<RelaxSimpleCodeGen>("relax.op.tanh"));
  map->emplace("where", std::make_shared<RelaxSimpleCodeGen>("relax.op.where"));

  // reduce axis ops
  map->emplace("argmax", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.argmax", false));
  map->emplace("argmin", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.argmin", false));
  map->emplace("max", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.max", true));
  map->emplace("min", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.min", true));
  map->emplace("mean", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.mean", true));
  map->emplace("sum", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.sum", true));
  map->emplace("prod", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.prod", true));
  map->emplace("std", std::make_shared<RelaxReduceAxisCodeGen>("relax.op.std", true));

  // axis && axes ops
  map->emplace("nn.log_softmax", std::make_shared<RelaxAxisCodeGen>("relax.op.nn.log_softmax"));
  map->emplace("nn.softmax", std::make_shared<RelaxAxisCodeGen>("relax.op.nn.softmax"));
  map->emplace("expand_dims", std::make_shared<RelaxAxesCodeGen>("relax.op.expand_dims"));
  map->emplace("squeeze", std::make_shared<RelaxAxesCodeGen>("relax.op.squeeze"));

  // math ops
  map->emplace("astype", std::make_shared<RelaxAstypeCodeGen>("relax.op.astype"));
  map->emplace("broadcast_to", std::make_shared<RelaxBroadcastToCodeGen>("relax.op.broadcast_to"));
  map->emplace("cast", std::make_shared<RelaxAstypeCodeGen>("relax.op.astype"));
  map->emplace("clip", std::make_shared<RelaxClipCodeGen>("relax.op.clip"));
  map->emplace("concat", std::make_shared<RelaxConcatCodeGen>("relax.op.concat"));
  map->emplace("concatenate", std::make_shared<RelaxConcatCodeGen>("relax.op.concat"));
  map->emplace("cumsum", std::make_shared<RelaxCumsumCodeGen>("relax.op.cumsum"));
  map->emplace("einsum", std::make_shared<RelaxEinsumCodeGen>("relax.op.einsum"));
  map->emplace("matmul", std::make_shared<RelaxMatmulCodeGen>("relax.op.linear_algebra.matmul"));
  map->emplace("permute_dims", std::make_shared<RelaxPermuteDimsCodeGen>("relax.op.permute_dims"));
  map->emplace("repeat", std::make_shared<RelaxRepeatCodeGen>("relax.op.repeat"));
  map->emplace("reshape", std::make_shared<RelaxReshapeCodeGen>("relax.op.reshape"));
  map->emplace("scatter_elements",
               std::make_shared<RelaxScatterElementsCodeGen>("relax.op.scatter_elements"));
  map->emplace("scatter_nd", std::make_shared<RelaxScatterNDCodeGen>("relax.op.scatter_nd"));
  map->emplace("split", std::make_shared<RelaxSplitCodeGen>("relax.op.split"));
  map->emplace("stack", std::make_shared<RelaxStackCodeGen>("relax.op.concat"));
  map->emplace("strided_slice",
               std::make_shared<RelaxStridedSliceCodeGen>("relax.op.strided_slice"));
  map->emplace("take", std::make_shared<RelaxTakeCodeGen>("relax.op.take"));
  map->emplace("tile", std::make_shared<RelaxTileCodeGen>("relax.op.tile"));
  map->emplace("transpose", std::make_shared<RelaxPermuteDimsCodeGen>("relax.op.permute_dims"));

  // create ops
  map->emplace("constant", std::make_shared<RelaxConstantCodeGen>("relax.Var"));
  map->emplace("full", std::make_shared<RelaxFullCodeGen>("relax.op.full"));
  map->emplace("ones", std::make_shared<RelaxCreateCodeGen>("relax.op.ones"));
  map->emplace("ones_like", std::make_shared<RelaxCreateLikeCodeGen>("relax.op.ones_like"));
  map->emplace("tril", std::make_shared<RelaxTriCodeGen>("relax.op.tril"));
  map->emplace("triu", std::make_shared<RelaxTriCodeGen>("relax.op.triu"));
  map->emplace("trilu", std::make_shared<RelaxTriCodeGen>(""));
  map->emplace("zeros", std::make_shared<RelaxCreateCodeGen>("relax.op.zeros"));
  map->emplace("zeros_like", std::make_shared<RelaxCreateLikeCodeGen>("relax.op.zeros_like"));

  // nn ops
  map->emplace("nn.adaptive_avg_pool2d",
               std::make_shared<RelaxAdaptivePool2dCodeGen>("relax.op.nn.adaptive_avg_pool2d"));
  map->emplace("nn.avg_pool2d", std::make_shared<RelaxPool2dCodeGen>("relax.op.nn.avg_pool2d"));
  map->emplace("nn.batch_matmul",
               std::make_shared<RelaxBatchMatmulCodeGen>("relax.op.linear_algebra.matmul"));
  map->emplace("nn.batch_norm", std::make_shared<RelaxBatchNormCodeGen>("relax.op.nn.batch_norm"));
  map->emplace("nn.bias_add", std::make_shared<RelaxBiasAddCodeGen>("relax.op.add"));
  map->emplace("nn.conv1d", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv1d", false));
  map->emplace("nn.conv2d", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv2d", false));
  map->emplace("nn.dense", std::make_shared<RelaxLinearCodeGen>("relax.op.linear_algebra.linear"));
  map->emplace("nn.gelu", std::make_shared<RelaxSimpleCodeGen>("relax.op.nn.gelu"));
  map->emplace("nn.group_norm", std::make_shared<RelaxGroupNormCodeGen>("relax.op.nn.group_norm"));
  map->emplace("nn.layer_norm", std::make_shared<RelaxLayerNormCodeGen>("relax.op.nn.layer_norm"));
  map->emplace("nn.max_pool2d", std::make_shared<RelaxPool2dCodeGen>("relax.op.nn.max_pool2d"));
  map->emplace("nn.nll_loss", std::make_shared<RelaxNllLossCodeGen>("relax.op.nn.nll_loss"));
  map->emplace("nn.pad", std::make_shared<RelaxPadCodeGen>("relax.op.nn.pad"));
  map->emplace("nn.relu", std::make_shared<RelaxSimpleCodeGen>("relax.op.nn.relu"));
  map->emplace("nn.silu", std::make_shared<RelaxSimpleCodeGen>("relax.op.nn.silu"));

  // image ops
  map->emplace("image.resize2d", std::make_shared<RelaxResize2dCodeGen>("relax.op.image.resize2d"));

  // special op
  map->emplace("get_item", std::make_shared<RelaxGetItemCodeGen>("relax.TupleGetItem"));
  map->emplace("shape", std::make_shared<RelaxShapeCodeGen>("relax.ShapeExpr"));
  map->emplace("tuple", std::make_shared<RelaxTupleCodeGen>("relax.Tuple"));
  map->emplace("plugin", std::make_shared<RelaxPluginOpCodeGen>("Plugin"));

  // msc ops
  map->emplace("msc.attention", std::make_shared<RelaxAttentionCodeGen>("relax.op.nn.attention"));
  map->emplace("msc.conv1d_bias", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv1d", true));
  map->emplace("msc.conv2d_bias", std::make_shared<RelaxConvCodeGen>("relax.op.nn.conv2d", true));
  map->emplace("msc.embedding", std::make_shared<RelaxEmbeddingCodeGen>("relax.op.take"));
  map->emplace("msc.gelu", std::make_shared<RelaxSimpleCodeGen>("relax.op.nn.gelu"));
  map->emplace("msc.linear",
               std::make_shared<RelaxLinearCodeGen>("relax.op.linear_algebra.linear"));
  map->emplace("msc.linear_bias",
               std::make_shared<RelaxLinearCodeGen>("relax.op.linear_algebra.linear"));
  map->emplace("msc.matmul",
               std::make_shared<RelaxMatmulCodeGen>("relax.op.linear_algebra.matmul"));

  return map;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
