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
#include "codegen_vsi_npu.h"

#include "../../../../runtime/contrib/vsi_npu/vsi_npu_runtime.h"
#include "../../utils.h"
#include "../codegen_c/codegen_c.h"
#include "op_map/op_setup.h"

#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <cassert>
#include <sstream>

#include "tim/transform/layout_inference.h"

namespace tvx = tim::vx;

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {

using TensorInfoTable = std::map<Expr, std::vector<tim::vx::TensorSpec>>;

void quant_info_infer(VxOpTable& op_tb, Expr now_expr, bool is_input) {
  auto now_opsetup = op_tb[now_expr];
  Expr pre_expr;
  if ((now_opsetup->pCallbackexpr_ == nullptr ||
      now_opsetup->pCallbackexpr_->ptr_pre_callback_ == nullptr) && is_input
      ) {
    return;
  } else if((now_opsetup->pCallbackexpr_ == nullptr ||
      now_opsetup->pCallbackexpr_->ptr_pre_callback_ == nullptr
      || op_tb[now_expr]->specs_[0].quantization_.ZeroPoints().size() == 0)&& !is_input ){
      return;
  } else {
    pre_expr = now_opsetup->pCallbackexpr_->ptr_pre_callback_->expr_;
  }

  auto pre_opsetup = op_tb[pre_expr];
  auto ptr_callback = pre_opsetup->pCallbackexpr_;

  if (now_opsetup->specs_[0].datatype_ == tvx::DataType::FLOAT32 ||
      pre_opsetup->specs_[0].datatype_ == tvx::DataType::FLOAT32 ||
      now_opsetup->specs_[0].datatype_ == tvx::DataType::BOOL8 ||
      pre_opsetup->specs_[0].datatype_ == tvx::DataType::BOOL8) {
    return;
  }

  tvx::Quantization& now_quant_info = now_opsetup->specs_[0].quantization_;

  std::vector<int32_t> zps;
  std::vector<float> scales;
  if (now_quant_info.Type() == tvx::QuantType::NONE) {
    zps = {0};
    scales = {1.0};
    now_quant_info.SetType(tvx::QuantType::ASYMMETRIC).SetScales({1.0}).SetZeroPoints({0});
  } else {
    zps = now_quant_info.ZeroPoints();
    scales = now_quant_info.Scales();
  }

  while (ptr_callback &&
         op_tb[ptr_callback->expr_]->specs_[0].quantization_.ZeroPoints().size() == 0) {
    Expr expr = ptr_callback->expr_;
    auto datatype = GetTvxType(expr->checked_type().as<TensorTypeNode>()->dtype);
    if (datatype != tim::vx::DataType::INT32) {
      op_tb[expr]
          ->specs_[0]
          .quantization_.SetType(tvx::QuantType::ASYMMETRIC)
          .SetScales(scales)
          .SetZeroPoints(zps);
    }
    ptr_callback = ptr_callback->ptr_pre_callback_;
  }
}

template <typename T, typename T2>
void attribute_transform(const T &attrs, T2 &attrs_num) {

  std::transform(attrs.begin(), attrs.end(), attrs_num.begin(),
                 [](const PrimExpr &attrs_num) {
                   return static_cast<uint32_t>(
                       attrs_num.as<IntImmNode>()->value);
                 });
};

std::shared_ptr<tvx::Tensor> createVxOPerand(TensorInfoTable tensor_info,
                                             Expr expr, tvx::Graph *graph,
                                             uint32_t idx = 0) {
  auto tensor_spec = tensor_info[expr][idx];
  void *data = expr->IsInstance<ConstantNode>()
                   ? expr.as<ConstantNode>()->data->data
                   : nullptr;
  return data == nullptr ? graph->CreateTensor(tensor_spec)
                         : graph->CreateTensor(tensor_spec, data);
};

static std::vector<tim::vx::TensorSpec>
GetTimVxTensorSpec(const TupleTypeNode *tuple) {
  auto input_node_tensors = tuple->fields;

  std::vector<tim::vx::TensorSpec> specs;
  uint32_t input_node_num = input_node_tensors.size();
  for (uint32_t i = 0; i < input_node_num; i++) {
    std::cout << "GetTimVxTensorSpec: " << input_node_tensors[i].as<TensorTypeNode>() << std::endl;
    tim::vx::ShapeType shape;
    std::transform(input_node_tensors[i].as<TensorTypeNode>()->shape.rbegin(),
                   input_node_tensors[i].as<TensorTypeNode>()->shape.rend(),
                   std::back_inserter(shape), [](const PrimExpr &dim) {
                     return static_cast<int>(dim.as<IntImmNode>()->value);
                   });

    auto dtype = input_node_tensors[i].as<TensorTypeNode>()->dtype;
    auto dataType = GetTvxType(dtype);

    tim::vx::TensorSpec spec(dataType, shape,
                             tim::vx::TensorAttribute::OUTPUT);
    specs.push_back(spec);
  }
  return specs;
}

using namespace backend;

std::map<Expr, std::shared_ptr<OpSetup>>
TensorMakerImpl::Create(const Expr &expr) {
  this->vxOpmap_tbl_.clear();
  CHECK(expr->checked_type().defined());
  if (auto tuple = expr->checked_type().as<TupleTypeNode>()) {
    auto specs = GetTimVxTensorSpec(tuple);
    auto tn = expr.as<TupleNode>();
    for (uint32_t i = 0; i < tuple->fields.size(); i++) {
      vxOpmap_tbl_[tn->fields[i]] = std::make_shared<OpSetup>(specs[i]);
    }
  }
  else {
    auto tensor_node = expr->checked_type().as<TensorTypeNode>();
    tim::vx::ShapeType o_shape;
    std::transform(tensor_node->shape.rbegin(), tensor_node->shape.rend(),
                   std::back_inserter(o_shape), [](const PrimExpr &dim) {
                     return static_cast<int>(dim.as<IntImmNode>()->value);
                   });

    auto dtype = tensor_node[0].dtype;
    auto tvx_type = GetTvxType(dtype);
    auto output_Opsetup = std::make_shared<OpSetup>(
        tvx::TensorSpec(tvx_type, o_shape, tvx::TensorAttribute::OUTPUT),
        std::make_shared<CallbackExpr>(expr));
    vxOpmap_tbl_[expr] = output_Opsetup;
  }
  VisitInferred(expr);
  return vxOpmap_tbl_;
}

typedef void (*setup_operand_fun_ptr)(VxOpTable&, Expr&);

template <typename T>
void setup_operand(VxOpTable& vxOpmap_tbl_, Expr& expr) {
  vxOpmap_tbl_[expr] = std::make_shared<T>(vxOpmap_tbl_[expr]->specs_[0],vxOpmap_tbl_[expr]->pCallbackexpr_);
}

#define DEFINE_NODE_ITEM(name, op) \
  {name, setup_operand<op>}

static std::map<std::string, setup_operand_fun_ptr> call_node_table = {
  DEFINE_NODE_ITEM("nn.relu", Relu),
  DEFINE_NODE_ITEM("nn.softmax", Softmax),
  DEFINE_NODE_ITEM("nn.avg_pool2d", AvgPool),
  DEFINE_NODE_ITEM("transpose", Transpose),
  DEFINE_NODE_ITEM("qnn.add", QnnAdd),
  DEFINE_NODE_ITEM("qnn.subtract", QnnSubtract),
  DEFINE_NODE_ITEM("qnn.mul", QnnMul),
  DEFINE_NODE_ITEM("maximum", Maximum),
  DEFINE_NODE_ITEM("minimum", Minimum),
  DEFINE_NODE_ITEM("nn.conv2d", Conv),
  DEFINE_NODE_ITEM("qnn.quantize", Quantize),
  DEFINE_NODE_ITEM("qnn.dequantize", Dequantize),
  DEFINE_NODE_ITEM("reshape", Reshape),
  DEFINE_NODE_ITEM("squeeze", Squeeze),
  DEFINE_NODE_ITEM("argmax", ArgMax),
  DEFINE_NODE_ITEM("argmin", ArgMin),
  DEFINE_NODE_ITEM("image.resize2d", Resize),
  DEFINE_NODE_ITEM("nn.max_pool2d", MaxPool2d),
  DEFINE_NODE_ITEM("qnn.concatenate", VsiNpuConcat),
  DEFINE_NODE_ITEM("add", Add),
  DEFINE_NODE_ITEM("mean", Mean),
  DEFINE_NODE_ITEM("sigmoid", Sigmoid),
  DEFINE_NODE_ITEM("tanh", Tanh),
  DEFINE_NODE_ITEM("nn.depth_to_space", DepthtoSpace),
  DEFINE_NODE_ITEM("logical_and", LogicalAnd),
  DEFINE_NODE_ITEM("logical_or", LogicalOr),
  DEFINE_NODE_ITEM("nn.pad", Pad),
  DEFINE_NODE_ITEM("nn.leaky_relu", LeakyRelu),
  DEFINE_NODE_ITEM("qnn.requantize", QnnRequantize),
};

static std::map<std::string, setup_operand_fun_ptr> func_node_table = {
  DEFINE_NODE_ITEM("vsi_npu.qnn_conv2d", VsiNpuQnnConv2d),
  DEFINE_NODE_ITEM("vsi_npu.qnn_avgpool2d", VsiNpuQnnAvgPool),
  DEFINE_NODE_ITEM("vsi_npu.qnn_softmax", VsiNpuQnnSoftmax),
  DEFINE_NODE_ITEM("vsi_npu.qnn_sigmoid", VsiNpuQnnSigmoid),
  DEFINE_NODE_ITEM("vsi_npu.qnn_clip", VsiNpuQnnClip),
  DEFINE_NODE_ITEM("vsi_npu.qnn_dense", VsiNpuQnnDense),
  DEFINE_NODE_ITEM("vsi_npu.qnn_mean", VsiNpuQnnMean),
  DEFINE_NODE_ITEM("vsi_npu.qnn_leaky_relu", VsiNpuQnnLeakyRelu),
  DEFINE_NODE_ITEM("vsi_npu.qnn_deconv", VsiNpuQnnDeconv),
  DEFINE_NODE_ITEM("vsi_npu.qnn_tanh", VsiNpuQnnTanh),
};

void TensorMakerImpl::InferCall(const CallNode *cn) {
  Call call_obj = GetRef<Call>(cn);
  Expr expr = GetRef<Expr>(cn);
  std::string name;
  tvx::Quantization out_quant = tvx::Quantization();
  if (const auto *fn = cn->op.as<FunctionNode>()) {
    auto comp = fn->GetAttr<String>(attr::kComposite);
    CHECK(comp.defined());
    name = comp.value();
    std::cout << "TensorMakerImpl::InferCall: " << name << std::endl;
    if (func_node_table.find(name) != func_node_table.end()) {
      func_node_table[name](vxOpmap_tbl_, expr);
      vxOpmap_tbl_[expr]->SetupOperand(cn, out_quant, vxOpmap_tbl_);
    }
  } else if (const auto *fn = cn->op.as<OpNode>()) {
    name = fn->name;
    std::cout << "TensorMakerImpl::InferCall: " << name << std::endl;
    if (call_node_table.find(name) != call_node_table.end()) {
      call_node_table[name](vxOpmap_tbl_, expr);
      vxOpmap_tbl_[expr]->SetupOperand(cn, out_quant, vxOpmap_tbl_);
    }
  } else {
    std::cout << __FUNCTION__ << "not support operator." << std::endl;
  }

  assert(vxOpmap_tbl_.find(expr) != vxOpmap_tbl_.end());
  auto &spec_info = vxOpmap_tbl_[expr]->specs_[0];
  if (out_quant.ZeroPoints().size() != 0) {
    spec_info.SetQuantization(out_quant);
  }
  quant_info_infer(vxOpmap_tbl_, expr, false);
}

void TensorMakerImpl::VisitInferred(const Expr &expr) {
  if (vxOpmap_tbl_.find(expr) != vxOpmap_tbl_.end() || expr->IsInstance<TupleNode>()) {
    VisitExpr(expr); // base class visit API
  }
}

void TensorMakerImpl::VisitExpr_(const CallNode *cn) {
  InferCall(cn);
  for (auto &arg : cn->args) {
    VisitInferred(arg);
  }
}

void TensorMakerImpl::VisitExpr_(const TupleNode *tn) {
  auto tuple = GetRef<Tuple>(tn);

  for (size_t i = 0; i < tn->fields.size(); i++) {
    if (vxOpmap_tbl_.find(tn->fields[i]) == vxOpmap_tbl_.end() &&
        vxOpmap_tbl_.find(tuple) != vxOpmap_tbl_.end()) {
      vxOpmap_tbl_[tn->fields[i]] =
          std::make_shared<OpSetup>(vxOpmap_tbl_[tuple]->specs_[i]);
    }
  }
  // Pre-order visitor
  for (const auto &field : tn->fields) {
    VisitExpr(field);
  }
}

std::shared_ptr<tvx::Context> GraphMakerImpl::vx_global_ctx_ =
    tvx::Context::Create(); // default construct

RawGraphDef GraphMakerImpl::Create(const Function &func) {
  std::cout << "GraphMakerImpl::Create" << std::endl;

  vx_graph_ = vx_global_ctx_->CreateGraph();
  vxOpmap_tbl_ = MakeTensor(this->module_, this->var_, func->body);
  std::vector<tvx::TensorSpec> input_spec, output_spec;
  for (const auto &param : func->params) {
    if (vxOpmap_tbl_.find(param) == vxOpmap_tbl_.end()) continue;
    quant_info_infer(vxOpmap_tbl_, param, true);
    for (auto &tensor_info : vxOpmap_tbl_[param]->specs_) {
      tensor_info.SetAttribute(tvx::TensorAttribute::INPUT);
      input_spec.push_back(tensor_info);
    }
  }
  VisitInferred(func->body);

  vx_graph_->PrintGraph();
  auto final_graph = tim::transform::LayoutInference(vx_graph_, vx_global_ctx_);
  final_graph.first->PrintGraph();
  size_t bin_size = -1;

  for (const auto& io_tensor : final_graph.first->OutputsTensor()) {
      output_spec.push_back(io_tensor->GetSpec());
  }

  bool is_ok = final_graph.first->CompileToBinary(nullptr, &bin_size);
  if (!is_ok) {
    std::cout << "Fatal error: compile to binary failed" << std::endl;
    assert(false);
  }
  assert(bin_size > 0 && is_ok);
  std::shared_ptr<char> nbg_buf(new char[bin_size]);
  is_ok = final_graph.first->CompileToBinary(nbg_buf.get(), &bin_size);
  assert(is_ok);

  RawGraphDef result;
  result.compiled_graph = nbg_buf;
  result.compiled_graph_size = bin_size;
  result.inputs_spec = input_spec;
  result.outputs_spec = output_spec;
  return result;
}

void GraphMakerImpl::VisitInferred(const Expr &expr) {
  if (vxOpmap_tbl_.find(expr) != vxOpmap_tbl_.end() || expr->IsInstance<TupleNode>()) {
    VisitExpr(expr);  // base class visit API
  }
}

void GraphMakerImpl::VisitExpr_(const CallNode *cn) {
  InferCall(cn);
  for (auto &arg : cn->args) {
    VisitInferred(arg);
  }
}

void GraphMakerImpl::VisitExpr_(const TupleNode *tn) {
  Tuple tuple = GetRef<Tuple>(tn);
  std::cout << "GraphMakerImpl::VisitExpr_(TupleNode): " << tn->fields.size() << std::endl;
  for (size_t i = 0; i < tn->fields.size(); i++) {
    if (vxOpmap_tbl_.find(tuple) != vxOpmap_tbl_.end() && vxOpmap_tbl_[tuple] != nullptr) {
      vxOpmap_tbl_[tn->fields[i]]->ptensors_ = {
          vxOpmap_tbl_[tuple]->ptensors_[i]};
    } else if (vxOpmap_tbl_[tn->fields[i]] != nullptr) {
      auto input =
          vx_graph_->CreateTensor(vxOpmap_tbl_[tn->fields[i]]->specs_[0]);
      vxOpmap_tbl_[tn->fields[i]]->SetTensor(input);
    }
  }
  for (const auto &arg : tuple->fields) {
    VisitInferred(arg);
  }
}

void GraphMakerImpl::InferCall(const CallNode *cn) {
  Call call = GetRef<Call>(cn);

  if (call->op->IsInstance<FunctionNode>()) {
    Function func = Downcast<Function>(call->op);
    CHECK(func.defined());
    auto name_node = func->GetAttr<String>(attr::kComposite);
    if (func_node_table.find(name_node.value()) != func_node_table.end()) {
      vxOpmap_tbl_[GetRef<Expr>(cn)]->SetupOperation(cn, vx_graph_,
                                                     vxOpmap_tbl_);
    }
  } else {
    Op op = Downcast<Op>(call->op);
    CHECK(op.defined());
    auto *op_node = cn->op.as<OpNode>();
    if (call_node_table.find(op_node->name) != call_node_table.end()) {
      vxOpmap_tbl_[GetRef<Expr>(cn)]->SetupOperation(cn, vx_graph_,
                                                     vxOpmap_tbl_);
    }
  }
}

tvm::runtime::Module VsiNpuCompiler::CreateRuntimeModule(const ObjectRef &ref) {
  RawGraphDef raw_graph;

  if (ref->IsInstance<FunctionNode>()) {
    IRModule mod;
    Function func = Downcast<Function>(ref);
    auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);

    CHECK(name_node.defined()) << "Failed to retrieved external symbol.";
    GlobalVar gvar = GlobalVar(name_node.value());
    std::cout << "This is important----> name_node.value() == "
              << name_node.value() << std::endl;
    mod->Add(gvar, func);
    Function mod_func = Downcast<Function>(mod->functions.at(gvar));

    raw_graph = MakeGraph(mod, gvar, mod_func);
  }

  return tvm::runtime::Module(make_object<tvm::runtime::vsi_npu::VsiNpuModule>(
      raw_graph.compiled_graph, raw_graph.compiled_graph_size,
      raw_graph.inputs_spec, raw_graph.outputs_spec));
}

} // namespace vsi_npu
} // namespace contrib
} // namespace relay
} // namespace tvm