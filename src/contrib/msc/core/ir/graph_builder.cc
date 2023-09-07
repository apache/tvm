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
 * \file src/contrib/msc/core/ir/graph_builder.cc
 */

#include "graph_builder.h"

namespace tvm {
namespace contrib {
namespace msc {

void RelaxFuncAttrGetter::VisitExpr_(const relax::CallNode* op) {
  if (op->attrs.defined()) {
    Map<String, String> attrs;
    AttrGetter getter(&attrs);
    const_cast<BaseAttrsNode*>(op->attrs.get())->VisitAttrs(&getter);
    for (const auto& pair : attrs) {
      if (attrs_.count(pair.first)) {
        int cnt = 1;
        String rep_key = pair.first;
        while (attrs_.count(rep_key + "_" + std::to_string(cnt))) {
          cnt++;
        }
        attrs_.Set(pair.first + "_" + std::to_string(cnt), pair.second);
      } else {
        attrs_.Set(pair.first, pair.second);
      }
    }
  }
}

const MSCGraph RelaxGraphBuilder::Build(const relax::Function& func) {
  // Add input nodes and record inputs;
  Array<String> input_names, output_names;
  for (const auto& p : func->params) {
    AddNode(p, NullOpt, p->name_hint());
    ICHECK(expr_tensor_map_.count(p)) << "Can not find func param " << p;
    input_names.push_back(expr_tensor_map_[p][0]);
  }
  VisitExpr(func);
  if (const auto* b_node = func->body.as<relax::SeqExprNode>()) {
    ICHECK(expr_tensor_map_.count(b_node->body)) << "Can not find seqexpr body " << b_node->body;
    output_names = expr_tensor_map_[b_node->body];
  } else {
    LOG(FATAL) << "Function body should be SeqExpr, get " << func->body;
  }
  // remove const nodes as weights
  Array<MSCJoint> valid_nodes;
  for (const auto& n : nodes_) {
    if (!weights_.count(n->name)) {
      n->index = valid_nodes.size();
      valid_nodes.push_back(n);
    }
  }
  const auto& graph = MSCGraph(name_, valid_nodes, input_names, output_names);
  // set inputs and outputs alias
  if (config_.input_aliases.size() == input_names.size()) {
    for (size_t i = 0; i < input_names.size(); i++) {
      graph->FindTensor(input_names[i])->alias = config_.input_aliases[i];
    }
  } else {
    for (size_t i = 0; i < input_names.size(); i++) {
      graph->FindTensor(input_names[i])->alias = graph->FindProducer(input_names[i])->name;
    }
  }
  if (config_.output_aliases.size() == output_names.size()) {
    for (size_t i = 0; i < output_names.size(); i++) {
      graph->FindTensor(output_names[i])->alias = config_.output_aliases[i];
    }
  } else {
    for (size_t i = 0; i < output_names.size(); i++) {
      const auto& output = graph->FindTensor(output_names[i]);
      if (output->alias.size() > 0) {
        continue;
      }
      const auto& producer = graph->FindProducer(output_names[i]);
      output->alias = producer->outputs.size() == 1
                          ? producer->name
                          : StringUtils::Replace(output_names[i], ":", "_");
    }
  }
  return graph;
}

const MSCJoint RelaxGraphBuilder::AddNode(const Expr& expr, const Optional<Expr>& binding_var,
                                          const String& name) {
  const auto& node_name = name.size() > 0 ? name : SpanUtils::GetAttr(expr->span, "name");
  const auto& shared_ref = SpanUtils::GetAttr(expr->span, "shared_ref");
  String optype;
  if (expr->IsInstance<relax::VarNode>()) {
    optype = "input";
  } else if (expr->IsInstance<relax::ConstantNode>()) {
    optype = "constant";
  } else if (expr->IsInstance<relax::ShapeExprNode>()) {
    optype = "shape";
  } else if (expr->IsInstance<relax::TupleGetItemNode>()) {
    optype = "get_item";
  } else if (expr->IsInstance<relax::TupleNode>()) {
    optype = "tuple";
  } else if (const auto* call_node = expr.as<relax::CallNode>()) {
    if (const auto* op_node = call_node->op.as<OpNode>()) {
      optype = StringUtils::Replace(op_node->name, "relax.", "");
    } else if (const auto* v_node = call_node->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<relax::Function>(ref_module_->Lookup(v_node->name_hint));
      const auto& name_opt = func->GetAttr<runtime::String>(relax::attr::kComposite);
      ICHECK(name_opt.defined()) << "Unexpected global func without composite";
      optype = name_opt.value();
    } else if (const auto* f_node = call_node->op.as<relax::FunctionNode>()) {
      const auto& name_opt = f_node->GetAttr<runtime::String>(relax::attr::kComposite);
      ICHECK(name_opt.defined()) << "Unexpected func without composite";
      optype = name_opt.value();
    } else {
      optype = "unknown_op";
    }
  } else {
    optype = "unknown_expr";
  }
  // Extract attributes
  Map<String, String> attrs;
  if (const auto* call_node = expr.as<relax::CallNode>()) {
    if (const auto* v_node = call_node->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<relax::Function>(ref_module_->Lookup(v_node->name_hint));
      attrs = RelaxFuncAttrGetter().GetAttrs(func);
    } else if (call_node->op->IsInstance<relax::FunctionNode>()) {
      attrs = RelaxFuncAttrGetter().GetAttrs(call_node->op);
    } else if (call_node->attrs.defined()) {
      AttrGetter getter(&attrs);
      const_cast<BaseAttrsNode*>(call_node->attrs.get())->VisitAttrs(&getter);
    }
  } else if (const auto* const_node = expr.as<relax::ConstantNode>()) {
    if (const_node->is_scalar()) {
      const float val = ExprUtils::GetScalar<float>(Downcast<relax::Constant>(expr));
      std::stringstream stream;
      stream << std::fixed << std::setprecision(config_.float_precision) << val;
      attrs.Set("scalar", stream.str());
    }
  } else if (const auto* shape_node = expr.as<relax::ShapeExprNode>()) {
    attrs.Set("shape", StringUtils::ToString(shape_node->values));
  } else if (const auto* get_node = expr.as<relax::TupleGetItemNode>()) {
    attrs.Set("index", std::to_string(get_node->index));
  }
  // Get scope
  Array<String> scope;
  if (optype != "input" && optype != "constant") {
    scope = StringUtils::Split(scope_name_, ".");
  }
  // Build inputs and weights
  Array<String> input_names;
  Map<String, MSCTensor> node_weights;
  if (const auto* call_node = expr.as<relax::CallNode>()) {
    const auto& input_types = ExprUtils::GetInputTypes(optype, call_node->args.size(), true);
    for (size_t i = 0; i < call_node->args.size(); i++) {
      const auto& arg = call_node->args[i];
      if (const auto* s_node = arg.as<relax::ShapeExprNode>()) {
        attrs.Set(input_types[i], StringUtils::ToString(s_node->values));
        continue;
      }
      if (const auto* s_node = arg.as<relax::PrimValueNode>()) {
        ICHECK(input_types[i] != "input") << i << " th PrimValue of " << optype
                                          << " should has special type, get " << input_types;
        attrs.Set(input_types[i], StringUtils::ToString(s_node->value));
        continue;
      }
      ICHECK(expr_tensor_map_.count(arg)) << "Missing argument " << arg;
      if (input_types[i] != "input" && arg->IsInstance<relax::ConstantNode>()) {
        const auto& t_name = expr_tensor_map_[arg][0];
        const auto& w_name = SpanUtils::GetAttr(arg->span, "name");
        const auto& pair = tensor_input_map_[t_name];
        const auto& producer = Downcast<MSCJoint>(pair.first);
        if (!weights_.count(w_name)) {
          const auto& ref = producer->OutputAt(pair.second);
          const auto& weight = MSCTensor(w_name, ref->dtype, ref->layout.name(), ref->shape);
          weights_.Set(w_name, weight);
        }
        if (producer->HasAttr("scalar")) {
          attrs.Set(input_types[i], producer->GetTypeAttr<std::string>("scalar"));
        }
        node_weights.Set(input_types[i], weights_[w_name]);
      } else {
        for (const auto& in_name : expr_tensor_map_[arg]) {
          input_names.push_back(in_name);
        }
      }
    }
  } else if (const auto* tuple_node = expr.as<relax::TupleNode>()) {
    for (const auto& f : tuple_node->fields) {
      ICHECK(expr_tensor_map_.count(f)) << "Can not find tuple field " << f;
      for (const auto& in_name : expr_tensor_map_[f]) {
        input_names.push_back(in_name);
      }
    }
  } else if (const auto* getitem_node = expr.as<relax::TupleGetItemNode>()) {
    ICHECK(expr_tensor_map_.count(getitem_node->tuple))
        << "Can not find tuple " << getitem_node->tuple;
    input_names = expr_tensor_map_[getitem_node->tuple];
  }
  std::vector<std::pair<BaseJoint, size_t>> inputs;
  for (const auto& i : input_names) {
    inputs.push_back(tensor_input_map_[i]);
  }
  // Build outputs
  Array<MSCTensor> outputs;
  const auto& layout = SpanUtils::GetAttr(expr->span, "layout");
  const auto& sinfo = relax::GetStructInfo(expr);
  if (const auto* t_info = sinfo.as<relax::TensorStructInfoNode>()) {
    const auto& opt_shape = t_info->GetShape();
    const auto& shape =
        opt_shape.defined() ? ArrayUtils::Cast<Integer>(opt_shape.value()) : Array<Integer>();
    const auto& output =
        MSCTensor(node_name + ":" + std::to_string(0), t_info->dtype, layout, shape);
    outputs.push_back(output);
  } else if (const auto* s_sinfo = sinfo.as<relax::ShapeStructInfoNode>()) {
    Array<Integer> shape{s_sinfo->ndim};
    const auto& output = MSCTensor(node_name + ":" + std::to_string(0),
                                   DataType(runtime::String2DLDataType("int32")), layout, shape);
    outputs.push_back(output);
  } else if (const auto* tuple_sinfo = sinfo.as<relax::TupleStructInfoNode>()) {
    Array<String> layouts = StringUtils::Split(layout, ",");
    if (layouts.size() == 0) {
      layouts = Array<String>(tuple_sinfo->fields.size(), "");
    }
    ICHECK_EQ(layouts.size(), tuple_sinfo->fields.size())
        << "Layout " << layout << " msimatch with fileds size " << tuple_sinfo->fields.size();
    size_t field_size = tuple_sinfo->fields.size();
    if (optype == "nn.batch_norm") {
      field_size = 1;
    }
    for (size_t i = 0; i < field_size; i++) {
      const auto& t_info = Downcast<relax::TensorStructInfo>(tuple_sinfo->fields[i]);
      const auto& opt_shape = t_info->GetShape();
      const auto& shape =
          opt_shape.defined() ? ArrayUtils::Cast<Integer>(opt_shape.value()) : Array<Integer>();
      const auto& output =
          MSCTensor(node_name + ":" + std::to_string(i), t_info->dtype, layouts[i], shape);
      outputs.push_back(output);
    }
  } else {
    LOG(FATAL) << "Unexpected struct info (" << sinfo->GetTypeKey() << ")" << sinfo;
  }
  // Build node
  const auto& node = MSCJoint(nodes_.size(), node_name, shared_ref, optype, attrs, scope, inputs,
                              outputs, node_weights);
  Array<String> output_names;
  for (size_t i = 0; i < outputs.size(); i++) {
    output_names.push_back(outputs[i]->name);
    tensor_input_map_[outputs[i]->name] = std::make_pair(node, i);
  }
  nodes_.push_back(node);
  const auto& ref_expr = binding_var.defined() ? binding_var.value() : expr;
  expr_tensor_map_.Set(ref_expr, output_names);
  return node;
}

void RelaxGraphBuilder::VisitBindingBlock(const relax::BindingBlock& block) {
  scope_name_ = SpanUtils::GetAttr(block->span, "name");
  RelaxExprVisitor::VisitBindingBlock(block);
}

void RelaxGraphBuilder::VisitExpr_(const relax::ConstantNode* op) {
  AddNode(GetRef<relax::Constant>(op));
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::ConstantNode* val) {
  AddNode(GetRef<relax::Constant>(val), binding->var);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::ShapeExprNode* val) {
  AddNode(GetRef<relax::ShapeExpr>(val), binding->var);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::CallNode* call_node) {
  RelaxExprVisitor::VisitBinding_(binding, call_node);
  try {
    AddNode(GetRef<relax::Call>(call_node), binding->var);
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to add node from " << binding->var << " : " << binding->value
                 << ", reason: " << err.message();
    throw err;
  }
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::TupleNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  AddNode(GetRef<relax::Tuple>(val), binding->var);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::TupleGetItemNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  AddNode(GetRef<relax::TupleGetItem>(val), binding->var);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::VarNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  const auto& output = GetRef<relax::Var>(val);
  ICHECK(expr_tensor_map_.count(output)) << "Can not find var " << output;
  expr_tensor_map_.Set(binding->var, expr_tensor_map_[output]);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::DataflowVarNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  const auto& output = GetRef<relax::DataflowVar>(val);
  ICHECK(expr_tensor_map_.count(output)) << "Can not find dataflow var " << output;
  expr_tensor_map_.Set(binding->var, expr_tensor_map_[output]);
}

Map<MSCTensor, NDArray> RelaxWeightsExtractor::GetWeights(const relax::Function& func) {
  VisitExpr(func);
  return weights_;
}

void RelaxWeightsExtractor::VisitExpr_(const relax::ConstantNode* op) {
  const auto& name = SpanUtils::GetAttr(op->span, "name");
  const auto& layout = SpanUtils::GetAttr(op->span, "layout");
  const auto& sinfo = relax::GetStructInfo(GetRef<relax::Constant>(op));
  ICHECK(sinfo->IsInstance<relax::TensorStructInfoNode>())
      << "Constant StrcutInfo should be TensorStructInfo";
  const auto& t_info = Downcast<relax::TensorStructInfo>(sinfo);
  const auto& opt_shape = t_info->GetShape();
  const auto& shape =
      opt_shape.defined() ? ArrayUtils::Cast<Integer>(opt_shape.value()) : Array<Integer>();
  const auto& weight = MSCTensor(name, t_info->dtype, layout, shape);
  weights_.Set(weight, op->data);
}

void RelayFuncAttrGetter::VisitExpr_(const relay::CallNode* op) {
  RelayExprVisitor::VisitExpr_(op);
  if (op->attrs.defined()) {
    Map<String, String> attrs;
    AttrGetter getter(&attrs);
    const_cast<BaseAttrsNode*>(op->attrs.get())->VisitAttrs(&getter);
    for (const auto& pair : attrs) {
      if (attrs_.count(pair.first)) {
        int cnt = 1;
        String rep_key = pair.first;
        while (attrs_.count(rep_key + "_" + std::to_string(cnt))) {
          cnt++;
        }
        attrs_.Set(pair.first + "_" + std::to_string(cnt), pair.second);
      } else {
        attrs_.Set(pair.first, pair.second);
      }
    }
  }
}

MSCGraph RelayGraphBuilder::Build(const relay::Function& func) {
  // Add input nodes and record inputs;
  Array<String> input_names, output_names;
  for (const auto& p : func->params) {
    AddNode(p, p->name_hint());
    ICHECK(expr_tensor_map_.count(p)) << "Can not find func param " << p;
    input_names.push_back(expr_tensor_map_[p][0]);
  }
  VisitExpr(func);
  ICHECK(expr_tensor_map_.count(func->body)) << "Can not find func body " << func->body;
  output_names = expr_tensor_map_[func->body];
  // remove const nodes as weights
  Array<MSCJoint> valid_nodes;
  for (const auto& n : nodes_) {
    if (!weights_.count(n->name)) {
      n->index = valid_nodes.size();
      valid_nodes.push_back(n);
    }
  }
  const auto& graph = MSCGraph(name_, valid_nodes, input_names, output_names);
  // set inputs and outputs alias
  if (config_.input_aliases.size() == input_names.size()) {
    for (size_t i = 0; i < input_names.size(); i++) {
      graph->FindTensor(input_names[i])->alias = config_.input_aliases[i];
    }
  } else {
    for (size_t i = 0; i < input_names.size(); i++) {
      graph->FindTensor(input_names[i])->alias = graph->FindProducer(input_names[i])->name;
    }
  }
  if (config_.output_aliases.size() == output_names.size()) {
    for (size_t i = 0; i < output_names.size(); i++) {
      graph->FindTensor(output_names[i])->alias = config_.output_aliases[i];
    }
  } else {
    for (size_t i = 0; i < output_names.size(); i++) {
      const auto& output = graph->FindTensor(output_names[i]);
      if (output->alias.size() > 0) {
        continue;
      }
      const auto& producer = graph->FindProducer(output_names[i]);
      output->alias = producer->outputs.size() == 1
                          ? producer->name
                          : StringUtils::Replace(output_names[i], ":", "_");
    }
  }
  return graph;
}

MSCJoint RelayGraphBuilder::AddNode(const Expr& expr, const String& name) {
  const auto& node_name = name.size() > 0 ? name : SpanUtils::GetAttr(expr->span, "name");
  const auto& shared_ref = SpanUtils::GetAttr(expr->span, "shared_ref");
  String optype;
  if (expr->IsInstance<relay::VarNode>()) {
    optype = "input";
  } else if (expr->IsInstance<relay::ConstantNode>()) {
    optype = "constant";
  } else if (expr->IsInstance<relay::TupleGetItemNode>()) {
    optype = "get_item";
  } else if (expr->IsInstance<relay::TupleNode>()) {
    optype = "tuple";
  } else if (const auto* call_node = expr.as<relay::CallNode>()) {
    if (const auto* op_node = call_node->op.as<OpNode>()) {
      optype = StringUtils::Replace(op_node->name, "relay.", "");
    } else {
      optype = "unknown_op";
    }
  } else if (const auto* f_node = expr.as<relay::FunctionNode>()) {
    const auto& name_opt = f_node->GetAttr<runtime::String>(relay::attr::kComposite);
    ICHECK(name_opt.defined()) << "Unexpected func without composite";
    optype = name_opt.value();
  } else {
    optype = "unknown_expr";
  }
  // Extract attributes
  Map<String, String> attrs;
  if (const auto* call_node = expr.as<relay::CallNode>()) {
    if (call_node->attrs.defined()) {
      AttrGetter getter(&attrs);
      const_cast<BaseAttrsNode*>(call_node->attrs.get())->VisitAttrs(&getter);
    }
  } else if (expr->IsInstance<relay::FunctionNode>()) {
    attrs = RelayFuncAttrGetter().GetAttrs(expr);
  } else if (const auto* const_node = expr.as<relay::ConstantNode>()) {
    if (const_node->is_scalar()) {
      const float val = ExprUtils::GetScalar<float>(Downcast<relay::Constant>(expr));
      std::stringstream stream;
      stream << std::fixed << std::setprecision(config_.float_precision) << val;
      attrs.Set("scalar", stream.str());
    }
  } else if (const auto* get_node = expr.as<relay::TupleGetItemNode>()) {
    attrs.Set("index", std::to_string(get_node->index));
  }
  // Get scope
  Array<String> scope;
  if (optype != "input" && optype != "constant") {
    scope.push_back("block");
  }
  // Build inputs and weights
  Array<String> input_names;
  Map<String, MSCTensor> node_weights;
  if (const auto* call_node = expr.as<relay::CallNode>()) {
    const auto& input_types = ExprUtils::GetInputTypes(optype, call_node->args.size(), false);
    for (size_t i = 0; i < call_node->args.size(); i++) {
      const auto& arg = call_node->args[i];
      ICHECK(expr_tensor_map_.count(arg)) << "Missing argument " << arg;
      if (input_types[i] != "input" && arg->IsInstance<relay::ConstantNode>()) {
        const auto& t_name = expr_tensor_map_[arg][0];
        const auto& w_name = SpanUtils::GetAttr(arg->span, "name");
        const auto& pair = tensor_input_map_[t_name];
        const auto& producer = Downcast<MSCJoint>(pair.first);
        if (!weights_.count(w_name)) {
          const auto& ref = producer->OutputAt(pair.second);
          const auto& weight = MSCTensor(w_name, ref->dtype, ref->layout.name(), ref->shape);
          weights_.Set(w_name, weight);
        }
        if (producer->HasAttr("scalar")) {
          attrs.Set(input_types[i], producer->GetTypeAttr<std::string>("scalar"));
        }
        node_weights.Set(input_types[i], weights_[w_name]);
      } else {
        for (const auto& in_name : expr_tensor_map_[arg]) {
          input_names.push_back(in_name);
        }
      }
    }
  } else if (const auto* f_node = expr.as<relay::FunctionNode>()) {
    for (const auto& p : f_node->params) {
      for (const auto& in_name : expr_tensor_map_[p]) {
        input_names.push_back(in_name);
      }
    }
    ICHECK(HasFuncScope()) << "Function without func scope " << relay::PrettyPrint(expr);
    const auto& weight_names = func_scopes_.top().GetFuncWeights();
    const auto& input_types =
        ExprUtils::GetInputTypes(optype, f_node->params.size() + weight_names.size(), false);
    for (size_t i = 0; i < weight_names.size(); i++) {
      const auto& pair = tensor_input_map_[weight_names[i]];
      const auto& producer = Downcast<MSCJoint>(pair.first);
      if (!weights_.count(producer->name)) {
        const auto& ref = producer->OutputAt(pair.second);
        const auto& weight = MSCTensor(producer->name, ref->dtype, ref->layout.name(), ref->shape);
        weights_.Set(producer->name, weight);
      }
      if (producer->HasAttr("scalar")) {
        attrs.Set(input_types[i], producer->GetTypeAttr<std::string>("scalar"));
      }
      node_weights.Set(input_types[i + f_node->params.size()], weights_[producer->name]);
    }
  } else if (const auto* tuple_node = expr.as<relay::TupleNode>()) {
    for (const auto& f : tuple_node->fields) {
      ICHECK(expr_tensor_map_.count(f)) << "Can not find tuple field " << f;
      for (const auto& in_name : expr_tensor_map_[f]) {
        input_names.push_back(in_name);
      }
    }
  } else if (const auto* getitem_node = expr.as<relay::TupleGetItemNode>()) {
    ICHECK(expr_tensor_map_.count(getitem_node->tuple))
        << "Can not find tuple " << getitem_node->tuple;
    input_names = expr_tensor_map_[getitem_node->tuple];
  }
  std::vector<std::pair<BaseJoint, size_t>> inputs;
  for (const auto& i : input_names) {
    inputs.push_back(tensor_input_map_[i]);
  }
  // Build outputs
  Array<MSCTensor> outputs;
  const auto& layout = SpanUtils::GetAttr(expr->span, "layout");
  Type checked_type = expr->checked_type_;
  if (checked_type.defined() && checked_type->IsInstance<relay::FuncTypeNode>()) {
    checked_type = Downcast<FuncType>(checked_type)->ret_type;
  }
  if (checked_type.defined()) {
    if (const auto* t_info = checked_type.as<relay::TensorTypeNode>()) {
      const auto& shape = ArrayUtils::Cast<Integer>(t_info->shape);
      const auto& output =
          MSCTensor(node_name + ":" + std::to_string(0), t_info->dtype, layout, shape);
      outputs.push_back(output);
    } else if (const auto* tuple_info = checked_type.as<relay::TupleTypeNode>()) {
      Array<String> layouts = StringUtils::Split(layout, ",");
      if (layouts.size() == 0) {
        layouts = Array<String>(tuple_info->fields.size(), "");
      }
      ICHECK_EQ(layouts.size(), tuple_info->fields.size())
          << "Layout " << layout << " msimatch with fileds size " << tuple_info->fields.size();
      size_t field_size = tuple_info->fields.size();
      if (optype == "nn.batch_norm") {
        field_size = 1;
      }
      for (size_t i = 0; i < field_size; i++) {
        const auto& t_info = Downcast<relay::TensorType>(tuple_info->fields[i]);
        const auto& shape = ArrayUtils::Cast<Integer>(t_info->shape);
        const auto& output =
            MSCTensor(node_name + ":" + std::to_string(i), t_info->dtype, layouts[i], shape);
        outputs.push_back(output);
      }
    } else {
      LOG(FATAL) << "Unexpected checked_type " << checked_type;
    }
  }

  // Build node
  const auto& node = MSCJoint(nodes_.size(), node_name, shared_ref, optype, attrs, scope, inputs,
                              outputs, node_weights);
  Array<String> output_names;
  for (size_t i = 0; i < outputs.size(); i++) {
    output_names.push_back(outputs[i]->name);
    tensor_input_map_[outputs[i]->name] = std::make_pair(node, i);
  }
  nodes_.push_back(node);
  expr_tensor_map_.Set(expr, output_names);
  return node;
}

void RelayGraphBuilder::VisitExpr_(const relay::ConstantNode* op) {
  const auto& node = AddNode(GetRef<relay::Constant>(op));
  if (HasFuncScope()) {
    func_scopes_.top().AddFuncWeight(node->OutputAt(0)->name);
  }
}

void RelayGraphBuilder::VisitExpr_(const relay::FunctionNode* op) {
  const auto& name_opt = op->GetAttr<runtime::String>(relay::attr::kComposite);
  if (name_opt.defined()) {
    StartFuncScope(SpanUtils::GetAttr(op->span, "name"));
  }
  RelayExprVisitor::VisitExpr_(op);
  if (HasFuncScope()) {
    AddNode(GetRef<relay::Function>(op));
    EndFuncScope();
  }
}

void RelayGraphBuilder::VisitExpr_(const relay::CallNode* op) {
  if (const auto* f_node = op->op.as<relay::FunctionNode>()) {
    const auto& name_opt = f_node->GetAttr<runtime::String>(relay::attr::kComposite);
    if (name_opt.defined()) {
      for (size_t i = 0; i < op->args.size(); i++) {
        ICHECK(expr_tensor_map_.count(op->args[i]))
            << "Can not find argument " << relay::PrettyPrint(op->args[i]);
        expr_tensor_map_.Set(f_node->params[i], expr_tensor_map_[op->args[i]]);
      }
    }
  }
  RelayExprVisitor::VisitExpr_(op);
  if (!HasFuncScope() && op->op->IsInstance<OpNode>()) {
    try {
      AddNode(GetRef<relay::Call>(op));
    } catch (runtime::InternalError& err) {
      LOG(WARNING) << "Failed to add node from " << relay::PrettyPrint(GetRef<relay::Call>(op))
                   << " : " << err.message();
      throw err;
    }
  }
  if (op->op->IsInstance<relay::FunctionNode>() && expr_tensor_map_.count(op->op)) {
    expr_tensor_map_.Set(GetRef<relay::Call>(op), expr_tensor_map_[op->op]);
  }
}

void RelayGraphBuilder::VisitExpr_(const relay::TupleNode* val) {
  RelayExprVisitor::VisitExpr_(val);
  AddNode(GetRef<relay::Tuple>(val));
}

void RelayGraphBuilder::VisitExpr_(const relay::TupleGetItemNode* val) {
  RelayExprVisitor::VisitExpr_(val);
  AddNode(GetRef<relay::TupleGetItem>(val));
}

void RelayGraphBuilder::StartFuncScope(const String& name) {
  RelayFuncScope func_scope = RelayFuncScope(name);
  func_scopes_.push(func_scope);
}
void RelayGraphBuilder::EndFuncScope() {
  ICHECK(HasFuncScope()) << "No FuncScope found";
  func_scopes_.pop();
}

bool RelayGraphBuilder::HasFuncScope() { return func_scopes_.size() > 0; }

Map<MSCTensor, NDArray> RelayWeightsExtractor::GetWeights(const relay::Function& func) {
  VisitExpr(func);
  return weights_;
}

void RelayWeightsExtractor::VisitExpr_(const relay::ConstantNode* op) {
  const auto& name = SpanUtils::GetAttr(op->span, "name");
  const auto& layout = SpanUtils::GetAttr(op->span, "layout");
  const auto& t_info = op->tensor_type();
  const auto& shape = ArrayUtils::Cast<Integer>(t_info->shape);
  const auto& weight = MSCTensor(name, t_info->dtype, layout, shape);
  weights_.Set(weight, op->data);
}

TVM_REGISTER_GLOBAL("msc.core.BuildFromRelax")
    .set_body_typed([](const IRModule& relax_module, const String& entry_name,
                       const String& options) -> MSCGraph {
      const auto& func = Downcast<relax::Function>(relax_module->Lookup(entry_name));
      return RelaxGraphBuilder(relax_module, entry_name, options).Build(func);
    });

TVM_REGISTER_GLOBAL("msc.core.GetRelaxWeights")
    .set_body_typed([](const IRModule& relax_module,
                       const String& entry_name) -> Map<MSCTensor, NDArray> {
      const auto& func = Downcast<relax::Function>(relax_module->Lookup(entry_name));
      return RelaxWeightsExtractor().GetWeights(func);
    });

TVM_REGISTER_GLOBAL("msc.core.BuildFromRelay")
    .set_body_typed([](const IRModule& relay_module, const String& entry_name,
                       const String& options) -> MSCGraph {
      const auto& func = Downcast<relay::Function>(relay_module->Lookup(entry_name));
      return RelayGraphBuilder(relay_module, entry_name, options).Build(func);
    });

TVM_REGISTER_GLOBAL("msc.core.GetRelayWeights")
    .set_body_typed([](const IRModule& relay_module,
                       const String& entry_name) -> Map<MSCTensor, NDArray> {
      const auto& func = Downcast<relay::Function>(relay_module->Lookup(entry_name));
      return RelayWeightsExtractor().GetWeights(func);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
