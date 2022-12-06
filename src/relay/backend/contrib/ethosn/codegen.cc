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
 * \file src/relay/backend/contrib/ethosn/codegen.cc
 * \brief The Relay -> Arm(R) Ethos(TM)-N command stream compiler.
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "codegen_ethosn.h"
#include "ethosn_api.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

constexpr size_t kReasonMaxLength = sl::g_ReasonMaxLength;

sl::TensorInfo GetTensorInfo(std::map<Expr, std::vector<sl::TensorInfo>> tensor_table,
                             const Call& call) {
  if (tensor_table.find(call) != tensor_table.end()) return tensor_table[call][0];

  return sl::TensorInfo();
}

bool IsEthosnOp(const Call& call, const std::string& op_name) {
  if (call->op->IsInstance<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    ICHECK(op.defined());
    return op == Op::Get(op_name);
  } else {
    return false;
  }
}

bool IsEthosnFunc(const Call& call, const std::string& op_name) {
  if (call->op->IsInstance<FunctionNode>()) {
    Function func = Downcast<Function>(call->op);
    ICHECK(func.defined());
    auto name_node = func->GetAttr<String>(attr::kComposite);
    return name_node.value() == op_name;
  }
  return false;
}

std::map<Expr, std::vector<sl::TensorInfo>> InferTensorsVisitor::Infer(const Expr& expr) {
  tensor_table_.clear();
  ICHECK(expr->checked_type().defined());
  size_t output_size = 1;
  if (auto tuple = expr->checked_type().as<TupleTypeNode>()) {
    output_size = tuple->fields.size();
  }
  for (size_t i = 0; i < output_size; i++) {
    tensor_table_[expr].push_back(sl::TensorInfo({1, 1, 1, 1}, sl::DataType::UINT8_QUANTIZED,
                                                 sl::DataFormat::NHWC, sl::QuantizationInfo()));
  }
  VisitInferred(expr);
  return tensor_table_;
}

void InferTensorsVisitor::InferCall(const CallNode* cn) {
  EthosnError err;
  Call call = GetRef<Call>(cn);
  // Determine call -> NPU mapping
  if (IsEthosnFunc(call, "ethos-n.qnn_conv2d")) {
    ConvolutionParams params;
    err += EthosnAPI::QnnConv2d(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_fc")) {
    FullyConnectedParams params;
    err += EthosnAPI::QnnFullyConnected(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnOp(call, "nn.max_pool2d")) {
    MaxPool2DParams params;
    params.input_info = GetTensorInfo(tensor_table_, call);
    err += EthosnAPI::MaxPool2D(call, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_avg_pool2d")) {
    AvgPool2DParams params;
    params.input_info = GetTensorInfo(tensor_table_, call);
    err += EthosnAPI::AvgPool2D(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnOp(call, "reshape")) {
    ReshapeParams params;
    params.input_info = GetTensorInfo(tensor_table_, call);
    err += EthosnAPI::Reshape(call, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_add")) {
    AdditionParams params;
    err += EthosnAPI::Addition(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.lhs_info};
    tensor_table_[cn->args[1]] = {params.rhs_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_sigmoid")) {
    SigmoidParams params;
    err += EthosnAPI::Sigmoid(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_mean")) {
    MeanParams params;
    err += EthosnAPI::Mean(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_tanh")) {
    TanhParams params;
    err += EthosnAPI::Tanh(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_leaky_relu")) {
    LeakyReLUParams params;
    err += EthosnAPI::LeakyReLU(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_conv2d_transpose")) {
    QnnConv2dTransposeParams params;
    err += EthosnAPI::QnnConv2dTranspose(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnOp(call, "qnn.concatenate")) {
    ConcatenateParams params;
    err = EthosnAPI::Concatenate(call, &params);
    tensor_table_[cn->args[0]] = params.input_infos;
  } else if (IsEthosnOp(call, "split")) {
    SplitParams params;
    params.input_info = GetTensorInfo(tensor_table_, call);
    err = EthosnAPI::Split(call, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnOp(call, "nn.depth_to_space")) {
    DepthToSpaceParams params;
    params.input_info = GetTensorInfo(tensor_table_, call);
    err += EthosnAPI::DepthToSpace(call, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnOp(call, "clip")) {
    ReluParams params;
    params.input_info = GetTensorInfo(tensor_table_, call);
    err = EthosnAPI::Relu(call, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_requantize")) {
    RequantizeParams params;
    err += EthosnAPI::Requantize(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_reinterpret_quantize")) {
    ReinterpretQuantizationParams params;
    err += EthosnAPI::ReinterpretQuantize(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else if (IsEthosnFunc(call, "ethos-n.qnn_resize")) {
    ResizeParams params;
    err += EthosnAPI::Resize(cn->op.as<FunctionNode>()->body, &params);
    tensor_table_[cn->args[0]] = {params.input_info};
  } else {
    err = EthosnError("unknown operator");
  }
  if (err) {
    ReportFatalError(call, err);
  }
}

// This will only visit an expression if the expression's tensor info
// has already been entirely inferred.
// An example where this is important is a tuple node where each
// get item node will only infer one field of the tuple's expression info.
// We don't want to traverse the tuple until all of its fields have been inferred.
void InferTensorsVisitor::VisitInferred(const Expr& expr) {
  if (tensor_table_.find(expr) != tensor_table_.end()) {
    for (const auto& tensor_info : tensor_table_[expr]) {
      if (tensor_info == sl::TensorInfo()) return;
    }
    VisitExpr(expr);
  }
}

void InferTensorsVisitor::VisitExpr_(const CallNode* cn) {
  InferCall(cn);
  // Pre-order visitor
  for (const auto& arg : cn->args) {
    VisitInferred(arg);
  }
}

void InferTensorsVisitor::VisitExpr_(const TupleNode* tn) {
  auto tuple = GetRef<Tuple>(tn);
  ICHECK(tensor_table_.find(tuple) != tensor_table_.end());
  for (size_t i = 0; i < tn->fields.size(); i++) {
    tensor_table_[tn->fields[i]] = {tensor_table_[tuple][i]};
  }
  // Pre-order visitor
  for (const auto& field : tn->fields) {
    VisitExpr(field);
  }
}

void InferTensorsVisitor::VisitExpr_(const TupleGetItemNode* tgn) {
  // Don't assume it must be targeting a TupleNode
  // Vars and calls can still have TupleType
  auto tg = GetRef<TupleGetItem>(tgn);
  ICHECK(tensor_table_.find(tg) != tensor_table_.end());
  auto tuple = tg->tuple;
  auto type = tuple->checked_type().as<TupleTypeNode>();
  int index = tg->index;
  // Resize the tensor infos to the tuple size if not already done
  if (tensor_table_.find(tuple) == tensor_table_.end()) {
    tensor_table_[tuple].resize(type->fields.size());
  }
  tensor_table_[tuple][index] = tensor_table_[tg][0];
  // Pre-order visitor
  VisitInferred(tuple);
}

sl::TensorsAndId MakeOps(const sl::TensorAndId<sl::Operand>& op) {
  sl::TensorsAndId ops;
  ops.tensors = {op.tensor};
  ops.operationId = op.operationId;
  return ops;
}

sl::EthosNVariant MakeVariant(EthosnCompilerConfig configuration) {
  String variant = configuration->variant;
  String tops = configuration->tops;
  String ple_ratio = configuration->ple_ratio;

  std::string capitalized_variant = variant;
  std::transform(capitalized_variant.begin(), capitalized_variant.end(),
                 capitalized_variant.begin(), ::toupper);
  std::string sl_variant_string =
      "Ethos-" + capitalized_variant + "_" + tops + "TOPS_" + ple_ratio + "PLE_RATIO";
  return sl::EthosNVariantFromString(sl_variant_string.c_str());
}

NetworkWithIDs ConstructNetworkVisitor::Construct(const Function& func) {
  // Initialise everything
  EthosnCompilerConfig cfg = GetCompilerAttrs();
  sl::EthosNVariant variant = MakeVariant(cfg);

  NetworkWithIDs network_with_ids;
  network_ = sl::CreateNetwork(
      sl::GetFwAndHwCapabilities(variant, static_cast<uint32_t>(std::stoul(cfg->sram_size))));
  network_with_ids.network = network_;
  operand_table_.clear();

  // Infer tensor information
  tensor_table_ = InferTensors(this->mod_, this->var_, func->body);
  // Add the inputs in the order they appear in the parameters
  unsigned int idx = 0;
  for (const auto& param : func->params) {
    for (const auto& tensor_info : tensor_table_[param]) {
      auto tensor_and_id = AddInput(network_, tensor_info);
      operand_table_[param].push_back(tensor_and_id.tensor);
      id_table_[param].push_back(std::make_pair(tensor_and_id.operationId, 0));
      network_with_ids.input_ids[tensor_and_id.operationId] = idx++;
    }
  }
  // Add the function body
  VisitExpr(func->body);
  // Add the outputs
  idx = 0;
  for (const auto& layer : operand_table_[func->body]) {
    AddOutput(network_, *layer);
    network_with_ids.output_ids[id_table_[func->body][idx]] = idx;
    idx++;
  }
  return network_with_ids;
}

sl::TensorsAndId ConstructNetworkVisitor::HandleCall(const CallNode* cn) {
  EthosnError err;
  Call call = GetRef<Call>(cn);
  sl::TensorAndId<sl::Operand> tensor;
  sl::TensorsAndId tensors;
  // Determine call -> NPU mapping
  if (IsEthosnFunc(call, "ethos-n.qnn_conv2d")) {
    if ((err = MakeConvolutionLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_fc")) {
    if ((err = MakeFullyConnectedLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnOp(call, "nn.max_pool2d")) {
    if ((err = MakeMaxPool2DLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_avg_pool2d")) {
    if ((err = MakeAvgPool2DLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnOp(call, "reshape")) {
    if ((err = MakeReshapeLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_add")) {
    if ((err = MakeAdditionLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_sigmoid")) {
    if ((err = MakeSigmoidLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_mean")) {
    if ((err = MakeMeanLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_tanh")) {
    if ((err = MakeTanhLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_leaky_relu")) {
    if ((err = MakeLeakyReLULayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_conv2d_transpose")) {
    if ((err = MakeConv2DTransposeLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnOp(call, "qnn.concatenate")) {
    if ((err = MakeConcatenateLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnOp(call, "split")) {
    if ((err = MakeSplitLayer(call, &tensors))) ReportFatalError(call, err);
    return tensors;
  } else if (IsEthosnOp(call, "nn.depth_to_space")) {
    if ((err = MakeDepthToSpaceLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnOp(call, "clip")) {
    if ((err = MakeReluLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_requantize")) {
    if ((err = MakeRequantizeLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_reinterpret_quantize")) {
    if ((err = MakeReinterpretQuantizeLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else if (IsEthosnFunc(call, "ethos-n.qnn_resize")) {
    if ((err = MakeResizeLayer(call, &tensor))) ReportFatalError(call, err);
    return MakeOps(tensor);
  } else {
    ReportFatalError(call, EthosnError("unknown operator"));
    return {};
  }
}

void ConstructNetworkVisitor::VisitExpr_(const CallNode* cn) {
  auto operand = HandleCall(cn);
  operand_table_[GetRef<Call>(cn)] = operand.tensors;
  for (size_t i = 0; i < operand.tensors.size(); i++) {
    id_table_[GetRef<Call>(cn)].push_back(std::make_pair(operand.operationId, i));
  }
}

void ConstructNetworkVisitor::VisitExpr_(const TupleNode* op) {
  Tuple tuple = GetRef<Tuple>(op);
  for (const auto& arg : tuple->fields) {
    // The fields in a tuple should not themselves be tuples
    // Nested tuples are not supported
    if (operand_table_[arg].size() == 1) {
      operand_table_[tuple].push_back(operand_table_[arg][0]);
      id_table_[tuple].push_back(id_table_[arg][0]);
    } else {
      operand_table_[tuple].push_back(nullptr);
      id_table_[tuple].push_back(std::make_pair(0, 0));
    }
  }
}

void ConstructNetworkVisitor::VisitExpr_(const TupleGetItemNode* tg) {
  Expr tuple = tg->tuple;
  operand_table_[GetRef<TupleGetItem>(tg)] = {operand_table_[tuple][tg->index]};
  id_table_[GetRef<TupleGetItem>(tg)] = {id_table_[tuple][tg->index]};
}

void ConstructNetworkVisitor::VisitLeaf(const Expr& expr) {
  // Don't traverse into functions, they're not supported
  if (!expr->IsInstance<FunctionNode>()) MixedModeVisitor::VisitLeaf(expr);
}

EthosnError ConstructNetworkVisitor::MakeConvolutionLayer(const Call& call,
                                                          sl::TensorAndId<sl::Operand>* out) {
  ConvolutionParams params;
  if (auto err = EthosnAPI::QnnConv2d(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto activation = operand_table_[call->args[0]][0];
  auto weights = AddConstant(network_, params.weights_info, params.raw_weights).tensor;
  auto bias = AddConstant(network_, params.bias_info, params.raw_bias).tensor;
  try {
    if (params.is_depthwise) {
      *out = AddDepthwiseConvolution(network_, *activation, *bias, *weights, params.conv_info);
    } else {
      *out = AddConvolution(network_, *activation, *bias, *weights, params.conv_info);
    }
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeFullyConnectedLayer(const Call& call,
                                                             sl::TensorAndId<sl::Operand>* out) {
  FullyConnectedParams params;
  if (auto err = EthosnAPI::QnnFullyConnected(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto weights = AddConstant(network_, params.weights_info, params.raw_weights->data).tensor;
  auto bias = AddConstant(network_, params.bias_info, params.raw_bias->data).tensor;
  try {
    auto input =
        AddReshape(network_, *operand_table_[call->args[0]][0], params.input_info.m_Dimensions)
            .tensor;
    *out = AddFullyConnected(network_, *input, *bias, *weights, params.fc_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeMaxPool2DLayer(const Call& call,
                                                        sl::TensorAndId<sl::Operand>* out) {
  MaxPool2DParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::MaxPool2D(call, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddPooling(network_, *input, params.pool_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeAvgPool2DLayer(const Call& call,
                                                        sl::TensorAndId<sl::Operand>* out) {
  AvgPool2DParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::AvgPool2D(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddPooling(network_, *input, params.pool_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeReshapeLayer(const Call& call,
                                                      sl::TensorAndId<sl::Operand>* out) {
  ReshapeParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::Reshape(call, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddReshape(network_, *input, params.new_shape);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeAdditionLayer(const Call& call,
                                                       sl::TensorAndId<sl::Operand>* out) {
  AdditionParams params;
  if (auto err = EthosnAPI::Addition(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto lhs = operand_table_[call->args[0]][0];
  auto rhs = operand_table_[call->args[1]][0];

  try {
    *out = AddAddition(network_, *lhs, *rhs, params.output_quantization_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeSigmoidLayer(const Call& call,
                                                      sl::TensorAndId<sl::Operand>* out) {
  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddSigmoid(network_, *input);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeMeanLayer(const Call& call,
                                                   sl::TensorAndId<sl::Operand>* out) {
  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddMeanXy(network_, *input);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeTanhLayer(const Call& call,
                                                   sl::TensorAndId<sl::Operand>* out) {
  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddTanh(network_, *input);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeLeakyReLULayer(const Call& call,
                                                        sl::TensorAndId<sl::Operand>* out) {
  LeakyReLUParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::LeakyReLU(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddLeakyRelu(network_, *input, params.leaky_relu_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeConv2DTransposeLayer(const Call& call,
                                                              sl::TensorAndId<sl::Operand>* out) {
  QnnConv2dTransposeParams params;
  if (auto err = EthosnAPI::QnnConv2dTranspose(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto activation = operand_table_[call->args[0]][0];
  auto weights = AddConstant(network_, params.weights_info, params.raw_weights->data).tensor;
  auto bias = AddConstant(network_, params.bias_info, params.raw_bias->data).tensor;
  try {
    *out = AddTransposeConvolution(network_, *activation, *bias, *weights, params.conv_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeConcatenateLayer(const Call& call,
                                                          sl::TensorAndId<sl::Operand>* out) {
  ConcatenateParams params;
  if (auto err = EthosnAPI::Concatenate(call, &params)) {
    return err;
  }

  std::vector<sl::Operand*> layers;
  auto ops = operand_table_[call->args[0]];

  for (const auto& op : ops) {
    layers.emplace_back(op.get());
  }
  try {
    *out = AddConcatenation(network_, layers, params.concat_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeSplitLayer(const Call& call, sl::TensorsAndId* outs) {
  SplitParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::Split(call, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *outs = AddSplit(network_, *input, params.split_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeDepthToSpaceLayer(const Call& call,
                                                           sl::TensorAndId<sl::Operand>* out) {
  DepthToSpaceParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::DepthToSpace(call, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddDepthToSpace(network_, *input, params.depth_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeReluLayer(const Call& call,
                                                   sl::TensorAndId<sl::Operand>* out) {
  ReluParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::Relu(call, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddRelu(network_, *input, params.relu_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeRequantizeLayer(const Call& call,
                                                         sl::TensorAndId<sl::Operand>* out) {
  RequantizeParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::Requantize(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddRequantize(network_, *input, params.requantize_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeReinterpretQuantizeLayer(
    const Call& call, sl::TensorAndId<sl::Operand>* out) {
  ReinterpretQuantizationParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::ReinterpretQuantize(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddReinterpretQuantization(network_, *input, params.reinterpret_quantize_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

EthosnError ConstructNetworkVisitor::MakeResizeLayer(const Call& call,
                                                     sl::TensorAndId<sl::Operand>* out) {
  ResizeParams params;
  params.input_info = GetTensorInfo(tensor_table_, call);
  if (auto err = EthosnAPI::Resize(call->op.as<FunctionNode>()->body, &params)) {
    return err;
  }

  auto input = operand_table_[call->args[0]][0];

  try {
    *out = AddResize(network_, *input, params.resize_info);
  } catch (const sl::NotSupportedException& e) {
    return EthosnError(e.what());
  }
  return EthosnError();
}

runtime::Module EthosnCompiler::CreateRuntimeModule(const ObjectRef& ref) {
  std::vector<runtime::ethosn::OrderedCompiledNetwork> cmms;
  if (ref->IsInstance<FunctionNode>()) {
    IRModule mod;
    Function func = Downcast<Function>(ref);
    auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(name_node.defined()) << "Failed to retrieved external symbol.";
    GlobalVar gvar = GlobalVar(name_node.value());
    mod->Add(gvar, func);
    Function mod_func = Downcast<Function>(mod->functions.at(gvar));
    cmms.emplace_back(CompileEthosnFunc(mod, gvar, mod_func));
  } else {
    LOG(FATAL) << "The input ref is expected to be a Relay function";
  }
  auto n = make_object<runtime::ethosn::EthosnModule>(&cmms);
  return runtime::Module(n);
}

runtime::ethosn::OrderedCompiledNetwork EthosnCompiler::CompileEthosnFunc(const IRModule& mod,
                                                                          const GlobalVar& gvar,
                                                                          const Function& func) {
  // Construct the network
  auto network_with_ids = ConstructNetwork(mod, gvar, func);
  // Now set the required build flags
  sl::CompilationOptions options = CreateOptions();
  // Finally compile the network
  std::vector<std::unique_ptr<sl::CompiledNetwork>> compiled_networks =
      sl::Compile(*network_with_ids.network, options);
  ICHECK_GE(compiled_networks.size(), 1) << "Ethos-N compiler failed to compile network";
  auto compiled_network = std::move(compiled_networks[0]);
  // Determine the order that the inputs/outputs are in and how that corresponds to the
  // order that the TVM runtime will expect them in
  auto input_output_order = GetInputOutputOrder(network_with_ids, compiled_network);
  auto io_sizes = GetIOSizes(compiled_network);
  // Use the order information to create an 'ordered' network with includes how to map
  // the inputs/outputs from the TVM runtime to the inputs/outputs of the compiled network
  runtime::ethosn::OrderedCompiledNetwork ordered_network;
  ordered_network.name = gvar->name_hint;
  ordered_network.compiled_cmm = std::move(compiled_network);
  ordered_network.inputs = input_output_order.first;
  ordered_network.outputs = input_output_order.second;
  ordered_network.input_sizes = io_sizes.first;
  ordered_network.output_sizes = io_sizes.second;
  return ordered_network;
}

sl::CompilationOptions EthosnCompiler::CreateOptions() {
  EthosnCompilerConfig cfg = GetCompilerAttrs();

  sl::CompilationOptions options;
  options.m_Strategy0 = cfg->strategy0;
  options.m_Strategy1 = cfg->strategy1;
  options.m_Strategy3 = cfg->strategy3;
  options.m_Strategy4 = cfg->strategy4;
  options.m_Strategy6 = cfg->strategy6;
  options.m_Strategy7 = cfg->strategy7;
  options.m_DebugInfo.m_DumpRam = cfg->dump_ram;
  options.m_DebugInfo.m_InitialSramDump = cfg->initial_sram_dump;
  options.m_BlockConfig16x16 = cfg->block_config_16x16;
  options.m_BlockConfig32x8 = cfg->block_config_32x8;
  options.m_BlockConfig8x32 = cfg->block_config_8x32;
  options.m_BlockConfig8x8 = cfg->block_config_8x8;
  options.m_EnableIntermediateCompression = cfg->enable_intermediate_compression;
  options.m_DisableWinograd = cfg->disable_winograd;
  options.m_DebugInfo.m_DebugDir = cfg->debug_dir;
  return options;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> EthosnCompiler::GetInputOutputOrder(
    NetworkWithIDs network, const std::unique_ptr<sl::CompiledNetwork>& compiled_network) {
  std::vector<sl::InputBufferInfo> input_infos = compiled_network->GetInputBufferInfos();
  std::vector<sl::OutputBufferInfo> output_infos = compiled_network->GetOutputBufferInfos();
  std::vector<uint32_t> input_order;
  std::vector<uint32_t> output_order;
  // Find the order of the inputs in the compiled network
  for (const auto& input_info : input_infos) {
    input_order.push_back(network.input_ids[input_info.m_SourceOperationId]);
  }
  // Find the order of the outputs in the compiled network
  for (const auto& output_info : output_infos) {
    auto output_id =
        std::make_pair(output_info.m_SourceOperationId, output_info.m_SourceOperationOutputIndex);
    output_order.push_back(network.output_ids[output_id]);
  }
  return std::make_pair(input_order, output_order);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> EthosnCompiler::GetIOSizes(
    const std::unique_ptr<sl::CompiledNetwork>& compiled_network) {
  std::vector<uint32_t> input_sizes;
  std::vector<uint32_t> output_sizes;
  for (const sl::InputBufferInfo info : compiled_network->GetInputBufferInfos()) {
    input_sizes.push_back(info.m_Size);
  }
  for (const sl::OutputBufferInfo info : compiled_network->GetOutputBufferInfos()) {
    output_sizes.push_back(info.m_Size);
  }

  return std::make_pair(input_sizes, output_sizes);
}

std::unique_ptr<sl::SupportQueries> EthosnCompiler::m_Queries;

EthosnError EthosnCompiler::SupportedSetup() {
  if (m_Queries == nullptr) {
    EthosnCompilerConfig cfg = GetCompilerAttrs();
    sl::EthosNVariant variant = MakeVariant(cfg);
    m_Queries = std::make_unique<sl::SupportQueries>(
        sl::GetFwAndHwCapabilities(variant, std::stoul(cfg->sram_size)));
    if (m_Queries == nullptr) {
      return EthosnError("Could not initialise Arm(R) Ethos(TM)-N compiler isSupported");
    }
  }
  return EthosnError();
}

TVM_REGISTER_GLOBAL("relay.ethos-n.support.conv2d")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      ConvolutionParams params;
      auto err = EthosnAPI::QnnConv2d(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      if (params.is_depthwise) {
        *rv = !err && EthosnCompiler::GetSupported()->IsDepthwiseConvolutionSupported(
                          params.bias_info, params.weights_info, params.conv_info,
                          params.input_info, &params.output_info, reason, sizeof(reason));
      } else {
        *rv = !err && EthosnCompiler::GetSupported()->IsConvolutionSupported(
                          params.bias_info, params.weights_info, params.conv_info,
                          params.input_info, &params.output_info, reason, sizeof(reason));
      }
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.fc")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      FullyConnectedParams params;
      auto err = EthosnAPI::QnnFullyConnected(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsFullyConnectedSupported(
                        params.bias_info, params.weights_info, params.fc_info, params.input_info,
                        &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.max_pool2d")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      MaxPool2DParams params;
      auto err = EthosnAPI::MaxPool2D(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err &&
            EthosnCompiler::GetSupported()->IsPoolingSupported(
                params.pool_info, params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.avg_pool2d")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      AvgPool2DParams params;
      auto err = EthosnAPI::AvgPool2D(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err &&
            EthosnCompiler::GetSupported()->IsPoolingSupported(
                params.pool_info, params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.reshape")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      ReshapeParams params;
      EthosnAPI::DefaultInputTensor(call);
      auto err = EthosnAPI::Reshape(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err &&
            EthosnCompiler::GetSupported()->IsReshapeSupported(
                params.new_shape, params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.addition")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      AdditionParams params;
      auto err = EthosnAPI::Addition(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsAdditionSupported(
                        params.lhs_info, params.rhs_info, params.output_quantization_info,
                        &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.sigmoid")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      SigmoidParams params;
      auto err = EthosnAPI::Sigmoid(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsSigmoidSupported(
                        params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.mean")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      MeanParams params;
      auto err = EthosnAPI::Mean(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsMeanXySupported(
                        params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.tanh")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      TanhParams params;
      auto err = EthosnAPI::Tanh(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsTanhSupported(
                        params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.leaky_relu")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      LeakyReLUParams params;
      auto err = EthosnAPI::LeakyReLU(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsLeakyReluSupported(
                        params.leaky_relu_info, params.input_info, &params.output_info, reason,
                        sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.conv2d_transpose")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      QnnConv2dTransposeParams params;
      auto err = EthosnAPI::QnnConv2dTranspose(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsTransposeConvolutionSupported(
                        params.bias_info, params.weights_info, params.conv_info, params.input_info,
                        &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.concatenate")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      ConcatenateParams params;
      auto err = EthosnAPI::Concatenate(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsConcatenationSupported(
                        params.input_infos, params.concat_info, &params.output_info, reason,
                        sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.split")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      SplitParams params;
      EthosnAPI::DefaultInputTensor(call);
      auto err = EthosnAPI::Split(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsSplitSupported(
                        params.input_info, params.split_info, nullptr, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.depth_to_space")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      DepthToSpaceParams params;
      auto err = EthosnAPI::DepthToSpace(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err &&
            EthosnCompiler::GetSupported()->IsDepthToSpaceSupported(
                params.input_info, params.depth_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.relu")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      ReluParams params;
      auto err = EthosnAPI::Relu(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err &&
            EthosnCompiler::GetSupported()->IsReluSupported(
                params.relu_info, params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.requantize")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      RequantizeParams params;
      auto err = EthosnAPI::Requantize(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsRequantizeSupported(
                        params.requantize_info, params.input_info, &params.output_info, reason,
                        sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.reinterpret_quantize")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      ReinterpretQuantizationParams params;
      auto err = EthosnAPI::ReinterpretQuantize(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err && EthosnCompiler::GetSupported()->IsReinterpretQuantizationSupported(
                        params.reinterpret_quantize_info, params.input_info, &params.output_info,
                        reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.support.resize")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      Call call = args[0];
      ResizeParams params;
      auto err = EthosnAPI::Resize(call, &params);
      err += EthosnCompiler::SupportedSetup();
      char reason[kReasonMaxLength];
      reason[0] = '\0';
      *rv = !err &&
            EthosnCompiler::GetSupported()->IsResizeSupported(
                params.resize_info, params.input_info, &params.output_info, reason, sizeof(reason));
      err += EthosnError(reason);
    });

TVM_REGISTER_GLOBAL("relay.ethos-n.query").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
#if defined ETHOSN_HW
  *rv = true;
#else
  *rv = false;
#endif
});

TVM_REGISTER_GLOBAL("relay.ethos-n.api.version").set_body_typed([]() -> String {
  return sl::GetLibraryVersion().ToString();
});

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
