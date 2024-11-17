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

#include <algorithm>
#include <set>

namespace tvm {
namespace contrib {
namespace msc {

const std::string GetScalarStr(const runtime::NDArray& data, int float_precision) {
  std::string scalar_str;
  if (data->dtype.code == kDLFloat) {
    const float val = ExprUtils::GetScalar<float>(data);
    std::stringstream stream;
    stream << std::fixed << std::setprecision(float_precision) << val;
    scalar_str = stream.str();
  } else {
    const int val = ExprUtils::GetScalar<int>(data);
    scalar_str = std::to_string(val);
  }
  return scalar_str;
}

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

void RelaxFuncAttrGetter::VisitExpr_(const relax::TupleGetItemNode* op) {
  attrs_.Set("index", std::to_string(op->index));
}

void RelaxFuncValueGetter::VisitExpr_(const relax::CallNode* op) {
  for (const auto& arg : op->args) {
    if (const auto* s_node = arg.as<relax::PrimValueNode>()) {
      values_.push_back(StringUtils::ToString(s_node->value));
    } else if (const auto* s_node = arg.as<relax::TupleNode>()) {
      bool all_values =
          std::all_of(s_node->fields.begin(), s_node->fields.end(),
                      [](const relax::Expr& e) { return e->IsInstance<relax::PrimValueNode>(); });
      if (all_values) {
        values_.push_back(StringUtils::ToString(s_node->fields));
      }
    }
  }
}

void RelaxFuncParamsFinder::VisitBinding_(const relax::VarBindingNode* binding,
                                          const relax::FunctionNode* val) {
  local_funcs_.Set(binding->var, GetRef<relax::Function>(val));
}

void RelaxFuncParamsFinder::VisitExpr_(const relax::CallNode* call_node) {
  RelaxExprVisitor::VisitExpr_(call_node);
  relax::Function func;
  if (const auto* v_node = call_node->op.as<GlobalVarNode>()) {
    func = Downcast<relax::Function>(ref_module_->Lookup(v_node->name_hint));
  } else if (call_node->op->IsInstance<relax::VarNode>()) {
    ICHECK(local_funcs_.count(call_node->op)) << "Can not find local func " << call_node->op;
    func = local_funcs_[call_node->op];
  }
  if (func.defined()) {
    for (size_t i = 0; i < call_node->args.size(); i++) {
      const auto& arg = call_node->args[i];
      if (arg->IsInstance<relax::VarNode>() && params_.count(Downcast<relax::Var>(arg))) {
        params_.Set(func->params[i], params_[Downcast<relax::Var>(arg)]);
      } else {
        params_.Set(func->params[i], arg);
      }
    }
  }
}

void RelaxLayoutsFinder::VisitBinding_(const relax::VarBindingNode* binding,
                                       const relax::FunctionNode* val) {
  local_funcs_.Set(binding->var, GetRef<relax::Function>(val));
}

void RelaxLayoutsFinder::VisitExpr_(const relax::CallNode* call_node) {
  RelaxExprVisitor::VisitExpr_(call_node);
  relax::Function func;
  if (const auto* v_node = call_node->op.as<GlobalVarNode>()) {
    func = Downcast<relax::Function>(ref_module_->Lookup(v_node->name_hint));
    VisitExpr(func);
  } else if (call_node->op->IsInstance<relax::VarNode>()) {
    ICHECK(local_funcs_.count(call_node->op)) << "Can not find local func " << call_node->op;
    func = local_funcs_[call_node->op];
  }
  if (func.defined()) {
    const auto& layouts_opt = func->GetAttr<runtime::Map<String, String>>(msc_attr::kInputLayouts);
    if (layouts_opt.defined()) {
      for (const auto& pair : layouts_opt.value()) {
        layouts_.Set(pair.first, pair.second);
      }
    }
  }
}

const MSCGraph RelaxGraphBuilder::Build(const relax::Function& func) {
  // Add input nodes and record inputs;
  Array<String> input_names, output_names;
  std::set<String> added_inputs;
  // Add prims
  for (const auto& p : func->params) {
    if (!p->struct_info_.defined()) {
      continue;
    }
    if (p->struct_info_.value()->IsInstance<relax::TensorStructInfoNode>()) {
      const auto& shape = ExprUtils::GetShape(p, false);
      for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i]->IsInstance<tvm::tir::VarNode>()) {
          Map<String, String> attrs;
          attrs.Set("producer", p->name_hint());
          attrs.Set("out_idx", "0");
          attrs.Set("dim", std::to_string(i));
          MatchOrCreatePrim(shape[i], "shape", Array<BaseJoint>(), attrs);
        }
      }
    } else {
      LOG_FATAL << "Unexpected func param " << p << "(" << p->GetTypeKey() << ")";
    }
  }

  for (const auto& p : func->params) {
    if (expr_tensor_map_.count(p)) {
      continue;
    }
    if (func_params_.count(p) && func_params_[p]->IsInstance<relax::ExternFuncNode>()) {
      continue;
    }
    if (func_params_.count(p) && func_params_[p]->IsInstance<relax::TupleNode>()) {
      const auto& tuple = Downcast<relax::Tuple>(func_params_[p]);
      Array<String> tuple_names;
      for (const auto& f : tuple->fields) {
        if (expr_tensor_map_.count(f)) {
          LOG_INFO << "Replica tuple input " << f;
        } else if (const auto* f_node = f.as<relax::VarNode>()) {
          AddNode(f, NullOpt, f_node->name_hint());
        } else {
          LOG_FATAL << "Unexpected tuple input " << f << "(" << f->GetTypeKey() << ")";
        }
        ICHECK(expr_tensor_map_.count(f)) << "Can not find func param from tuple " << f;
        for (const auto& name : expr_tensor_map_[f]) {
          tuple_names.push_back(name);
        }
      }
      expr_tensor_map_.Set(p, tuple_names);
    } else {
      AddNode(p, NullOpt, p->name_hint());
    }
    ICHECK(expr_tensor_map_.count(p)) << "Can not find func param " << p;
    for (const auto& name : expr_tensor_map_[p]) {
      if (!added_inputs.count(name)) {
        input_names.push_back(name);
        added_inputs.insert(name);
      }
    }
  }
  VisitExpr(func);
  ICHECK(expr_tensor_map_.count(func->body->body))
      << "Can not find seqexpr body " << func->body->body;
  output_names = expr_tensor_map_[func->body->body];
  // remove const nodes as weights
  Array<MSCJoint> valid_nodes;
  std::set<String> ignore_inputs;
  for (const auto& n : nodes_) {
    if (weights_.count(n->name) || ignore_nodes_.count(n->name)) {
      for (const auto& o : n->outputs) {
        ignore_inputs.insert(o->name);
      }
    } else {
      n->index = valid_nodes.size();
      valid_nodes.push_back(n);
      if (n->optype != "input") {
        for (const auto& o : n->outputs) {
          ignore_inputs.insert(o->name);
        }
      }
    }
  }
  // remove uselese inputs
  Array<String> valid_inputs;
  for (const auto& i : input_names) {
    if (!ignore_inputs.count(i)) {
      valid_inputs.push_back(i);
    }
  }
  // build graph
  const auto& graph = MSCGraph(name_, valid_nodes, valid_inputs, output_names, prims_);
  // set inputs and outputs alias
  if (config_.input_aliases.size() == valid_inputs.size()) {
    for (size_t i = 0; i < valid_inputs.size(); i++) {
      graph->FindTensor(valid_inputs[i])->alias = config_.input_aliases[i];
    }
  } else {
    for (size_t i = 0; i < valid_inputs.size(); i++) {
      graph->FindTensor(valid_inputs[i])->alias = graph->FindProducer(valid_inputs[i])->name;
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
  // Get optype, node_name and layout
  String node_name = name.size() > 0 ? name : SpanUtils::GetAttr(expr->span, msc_attr::kName);
  String optype = "unknown";
  String layout = SpanUtils::GetAttr(expr->span, msc_attr::kLayout);
  if (func_params_.count(expr) && func_params_[expr]->IsInstance<relax::ConstantNode>()) {
    node_name = SpanUtils::GetAttr(func_params_[expr]->span, msc_attr::kName);
    optype = "constant";
  } else if (expr->IsInstance<relax::VarNode>()) {
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
      if (op_node->name == "relax.call_dps_packed") {
        optype = Downcast<relax::ExternFunc>(call_node->args[0])->global_symbol;
      } else {
        optype = StringUtils::Replace(op_node->name, "relax.", "");
      }
    } else if (const auto* v_node = call_node->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<relax::Function>(ref_module_->Lookup(v_node->name_hint));
      std::tie(node_name, optype, layout) = ParseFunc(func);
    } else if (call_node->op->IsInstance<relax::VarNode>()) {
      ICHECK(target_funcs_.count(call_node->op)) << "Can not find target func: " << call_node->op;
      std::tie(node_name, optype, layout) = ParseFunc(target_funcs_[call_node->op]);
    } else if (call_node->op->IsInstance<relax::FunctionNode>()) {
      std::tie(node_name, optype, layout) = ParseFunc(Downcast<relax::Function>(call_node->op));
    }
  }
  if (layouts_.count(node_name)) {
    layout = layouts_[node_name];
  }

  // specail case for tuple
  if (optype == "tuple" && expr->IsInstance<relax::CallNode>() &&
      Downcast<relax::Call>(expr)->op->IsInstance<relax::VarNode>()) {
    const auto& call_node = Downcast<relax::Call>(expr);
    ICHECK(target_funcs_.count(call_node->op)) << "Can not find target func: " << call_node->op;
    const auto& tuple_func = target_funcs_[call_node->op];
    for (size_t i = 0; i < call_node->args.size(); i++) {
      expr_tensor_map_.Set(tuple_func->params[i], expr_tensor_map_[call_node->args[i]]);
    }
    VisitExpr(tuple_func);
    ICHECK(expr_tensor_map_.count(tuple_func->body->body))
        << "Can not find seqexpr body " << tuple_func->body->body;
    const auto& outputs = expr_tensor_map_[tuple_func->body->body];
    const auto& ref_expr = binding_var.defined() ? binding_var.value() : expr;
    expr_tensor_map_.Set(ref_expr, outputs);
    ICHECK(tensor_input_map_.count(outputs[0])) << "Can not find tensor " << outputs[0];
    return Downcast<MSCJoint>(tensor_input_map_[outputs[0]].first);
  }

  // get plugin
  const auto& plugin = IsPlugin(optype) ? GetPlugin(optype) : Plugin();

  // Extract normal attributes
  Map<String, String> attrs;
  if (plugin.defined()) {
    const auto& op = Downcast<relax::Call>(expr)->op;
    if (target_funcs_.count(op)) {
      const auto& opattrs_opt = target_funcs_[op]->GetAttr<Array<String>>(msc_attr::kOpattrs);
      if (opattrs_opt.defined()) {
        const auto& opattrs = opattrs_opt.value();
        ICHECK_EQ(opattrs.size(), plugin->attrs.size())
            << "opattrs " << opattrs << " size mismatch with " << plugin->attrs.size();
        for (size_t i = 0; i < opattrs.size(); i++) {
          attrs.Set(plugin->attrs[i]->name, opattrs[i]);
        }
      }
    } else {
      const auto& args = GetPluginInputs(expr);
      for (size_t i = 0; i < plugin->attrs.size(); i++) {
        const auto& val = args[plugin->inputs.size() + i];
        attrs.Set(plugin->attrs[i]->name, StringUtils::ToString(val));
      }
    }
  } else if (const auto* call_node = expr.as<relax::CallNode>()) {
    if (const auto* v_node = call_node->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<relax::Function>(ref_module_->Lookup(v_node->name_hint));
      const auto& name_opt = func->GetAttr<runtime::String>(relax::attr::kComposite);
      if (name_opt.defined()) {
        attrs = RelaxFuncAttrGetter().GetAttrs(func);
      }
    } else if (call_node->op->IsInstance<relax::VarNode>()) {
      ICHECK(target_funcs_.count(call_node->op)) << "Can not find target func: " << call_node->op;
      attrs = RelaxFuncAttrGetter().GetAttrs(target_funcs_[call_node->op]);
    } else if (call_node->op->IsInstance<relax::FunctionNode>()) {
      attrs = RelaxFuncAttrGetter().GetAttrs(call_node->op);
    } else if (call_node->attrs.defined()) {
      AttrGetter getter(&attrs);
      const_cast<BaseAttrsNode*>(call_node->attrs.get())->VisitAttrs(&getter);
    }
  } else if (const auto* const_node = expr.as<relax::ConstantNode>()) {
    if (const_node->is_scalar()) {
      attrs.Set("scalar", GetScalarStr(const_node->data, config_.float_precision));
    }
  } else if (const auto* shape_node = expr.as<relax::ShapeExprNode>()) {
    attrs.Set("shape", StringUtils::ToString(shape_node->values));
  } else if (const auto* get_node = expr.as<relax::TupleGetItemNode>()) {
    attrs.Set("index", std::to_string(get_node->index));
  }

  // Extract attributes from arguments
  Array<String> input_types;
  if (!plugin.defined() && expr->IsInstance<relax::CallNode>()) {
    const auto& call = Downcast<relax::Call>(expr);
    Array<String> values;
    if (call->op->IsInstance<relax::VarNode>()) {
      ICHECK(target_funcs_.count(call->op)) << "Can not find target func: " << call->op;
      values = RelaxFuncValueGetter().GetValues(target_funcs_[call->op]);
    }
    input_types = ExprUtils::GetInputTypes(optype, call->args.size() + values.size(), true);
    for (size_t i = 0; i < call->args.size(); i++) {
      const auto& arg = call->args[i];
      if (const auto* s_node = arg.as<relax::ShapeExprNode>()) {
        attrs.Set(input_types[i], StringUtils::ToString(s_node->values));
      } else if (func_params_.count(arg) && func_params_[arg]->IsInstance<relax::ShapeExprNode>()) {
        const auto* s_node = func_params_[arg].as<relax::ShapeExprNode>();
        attrs.Set(input_types[i], StringUtils::ToString(s_node->values));
        ignore_nodes_.insert(Downcast<relax::Var>(arg)->name_hint());
      } else if (const auto* s_node = arg.as<relax::PrimValueNode>()) {
        ICHECK(input_types[i] != "input") << i << " th PrimValue of " << optype
                                          << " should has special type, get " << input_types;
        attrs.Set(input_types[i], StringUtils::ToString(s_node->value));
      } else if (input_types[i] != "input" && arg->IsInstance<relax::TupleNode>()) {
        attrs.Set(input_types[i], StringUtils::ToString(arg));
      }
    }
    for (size_t i = call->args.size(); i < input_types.size(); i++) {
      attrs.Set(input_types[i], values[i - call->args.size()]);
    }
  }

  // Build inputs and weights
  Array<String> input_names;
  Map<String, MSCTensor> node_weights;
  if (plugin.defined()) {
    const auto& call = Downcast<relax::Call>(expr);
    if (call->args.size() == 1) {
      ICHECK(expr_tensor_map_.count(call->args[0]))
          << "Can not find tuple plugin input " << call->args[0];
      input_names = expr_tensor_map_[call->args[0]];
    } else {
      const auto& args = GetPluginInputs(expr);
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        ICHECK(expr_tensor_map_.count(args[i])) << "Can not find plugin input " << args[i];
        for (const auto& in_name : expr_tensor_map_[args[i]]) {
          input_names.push_back(in_name);
        }
      }
    }
  } else if (const auto* call_node = expr.as<relax::CallNode>()) {
    for (size_t i = 0; i < call_node->args.size(); i++) {
      if (attrs.count(input_types[i])) {
        continue;
      }
      const auto& arg = call_node->args[i];
      Array<String> arg_names;
      if (expr_tensor_map_.count(arg)) {
        arg_names = expr_tensor_map_[arg];
      } else if (input_types[i] == "input" && arg->IsInstance<relax::TupleNode>()) {
        const auto* tuple_node = arg.as<relax::TupleNode>();
        for (const auto& f : tuple_node->fields) {
          ICHECK(expr_tensor_map_.count(f)) << "Can not find tuple field " << f;
          for (const auto& in_name : expr_tensor_map_[f]) {
            arg_names.push_back(in_name);
          }
        }
      }
      String weight_name;
      if (input_types[i] != "input" && arg->IsInstance<relax::ConstantNode>()) {
        weight_name = SpanUtils::GetAttr(arg->span, msc_attr::kName);
      } else if (input_types[i] != "input" && func_params_.count(arg) &&
                 func_params_[arg]->IsInstance<relax::ConstantNode>()) {
        weight_name = SpanUtils::GetAttr(func_params_[arg]->span, msc_attr::kName);
        ignore_nodes_.insert(Downcast<relax::Var>(arg)->name_hint());
      }
      // set weights or inputs
      if (weight_name.size() > 0) {
        const auto& t_name = arg_names[0];
        const auto& pair = tensor_input_map_[t_name];
        const auto& producer = Downcast<MSCJoint>(pair.first);
        if (!weights_.count(weight_name)) {
          const auto& ref = producer->OutputAt(pair.second);
          MSCTensor weight;
          if (input_types[i] == "bias") {
            weight = MSCTensor(weight_name, ref->dtype, "O", Array<Integer>{ref->GetSize()});
          } else if (input_types[i] == "weight" &&
                     (optype == "msc.linear" || optype == "msc.linear_bias")) {
            if (ref->layout.name() == "IO") {
              String valid_layout = ref->layout[1].name() + ref->layout[0].name();
              const auto& valid_shape = Array<Integer>({ref->shape[1], ref->shape[0]});
              weight = MSCTensor(weight_name, ref->dtype, valid_layout, valid_shape);
            } else {
              weight = MSCTensor(weight_name, ref->dtype, ref->layout.name(), ref->shape);
            }
          } else {
            weight = MSCTensor(weight_name, ref->dtype, ref->layout.name(), ref->shape);
          }
          weights_.Set(weight_name, weight);
        }
        if (producer->HasAttr("scalar")) {
          attrs.Set(input_types[i], producer->GetTypeAttr<std::string>("scalar"));
        }
        node_weights.Set(input_types[i], weights_[weight_name]);
      } else {
        for (const auto& in_name : arg_names) {
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
  } else if (optype == "constant") {
    const auto& t_info = Downcast<relax::TensorStructInfo>(relax::GetStructInfo(expr));
    const auto& shape_opt = t_info->GetShape();
    ICHECK(shape_opt.defined()) << "Constant shape is not defined";
    const auto& weight =
        MSCTensor(node_name, t_info->dtype, layout, ArrayUtils::Cast<Integer>(shape_opt.value()));
    node_weights.Set("const", weight);
  }
  std::vector<std::pair<BaseJoint, size_t>> inputs;
  for (const auto& i : input_names) {
    inputs.push_back(tensor_input_map_[i]);
  }

  // Redefine layout for special ops
  if (optype == "tuple") {
    layout = "";
    for (size_t i = 0; i < inputs.size(); i++) {
      const auto& in_tensor = Downcast<MSCJoint>(inputs[i].first)->OutputAt(inputs[i].second);
      layout = layout + in_tensor->layout.name();
      layout = layout + (i == inputs.size() - 1 ? "" : ",");
    }
  } else if (optype == "get_item") {
    int idx = std::stoi(attrs["index"]);
    const auto& in_tensor = Downcast<MSCJoint>(inputs[idx].first)->OutputAt(inputs[idx].second);
    layout = in_tensor->layout.name();
  }

  // Build output tensor
  auto build_output = [this](const relax::StructInfo& sinfo, const String& node_name,
                             const String& layout) {
    ICHECK(sinfo->IsInstance<relax::TensorStructInfoNode>())
        << "sinfo should be TensorStructInfo, get " << sinfo->GetTypeKey();
    const auto& t_info = Downcast<relax::TensorStructInfo>(sinfo);
    const auto& shape = ArrayUtils::Cast<Integer>(ExprUtils::GetShape(t_info));
    Array<String> prims;
    bool has_prims = false;
    if (shape.size() > 0) {
      for (const auto& s : t_info->GetShape().value()) {
        if (prim_map_.count(s)) {
          prims.push_back(prim_map_[s]->name);
          has_prims = true;
        } else {
          prims.push_back(StringUtils::ToString(s));
        }
      }
    }
    if (has_prims) {
      return MSCTensor(node_name, t_info->dtype, layout, shape, "", prims);
    }
    return MSCTensor(node_name, t_info->dtype, layout, shape);
  };

  // Gather outputs
  Array<MSCTensor> outputs;
  const auto& sinfo = relax::GetStructInfo(expr);
  Array<String> layouts = StringUtils::Split(layout, ",");
  size_t num_output = 1;
  if (const auto* tuple_sinfo = sinfo.as<relax::TupleStructInfoNode>()) {
    num_output = tuple_sinfo->fields.size();
  }
  if (layouts.size() == 0) {
    layouts = Array<String>(num_output, "");
  }
  ICHECK_EQ(layouts.size(), num_output)
      << "Layouts " << layouts << " msimatch with output size " << num_output;
  if (sinfo->IsInstance<relax::TensorStructInfoNode>()) {
    const auto& t_name = node_name + ":" + std::to_string(0);
    outputs.push_back(build_output(sinfo, t_name, layouts[0]));
  } else if (const auto* s_sinfo = sinfo.as<relax::ShapeStructInfoNode>()) {
    Array<Integer> shape{s_sinfo->ndim};
    const auto& t_name = node_name + ":" + std::to_string(0);
    const auto& dtype = DataType(runtime::String2DLDataType("int32"));
    outputs.push_back(MSCTensor(t_name, dtype, layouts[0], shape));
  } else if (const auto* tuple_sinfo = sinfo.as<relax::TupleStructInfoNode>()) {
    size_t field_size = optype == "nn.batch_norm" ? 1 : num_output;
    for (size_t i = 0; i < field_size; i++) {
      const auto& t_name = node_name + ":" + std::to_string(i);
      outputs.push_back(build_output(tuple_sinfo->fields[i], t_name, layouts[i]));
    }
  } else {
    LOG(FATAL) << "Unexpected struct info (" << sinfo->GetTypeKey() << ")" << sinfo;
  }

  // Build node
  Array<String> scope;
  if (optype != "input" && optype != "constant") {
    scope = StringUtils::Split(scope_name_, ".");
  }
  const auto& shared_ref = SpanUtils::GetAttr(expr->span, msc_attr::kSharedRef);
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
  String block_name = SpanUtils::GetAttr(block->span, msc_attr::kName);
  if (block_name.size() == 0) {
    block_name = "block";
  }
  const String& prefix = StringUtils::Join(block_stack_, ".");
  if (setted_blocks_.count(prefix + "." + block_name)) {
    int cnt = 1;
    while (setted_blocks_.count(prefix + "." + block_name + "_" + std::to_string(cnt))) {
      cnt++;
    }
    block_name = block_name + "_" + std::to_string(cnt);
  }
  scope_name_ = prefix + "." + block_name;
  setted_blocks_.insert(scope_name_);
  block_stack_.push_back(block_name);
  RelaxExprVisitor::VisitBindingBlock(block);
  block_stack_.pop_back();
}

#define ADD_BINARY_PRIM(TypeName)                                                 \
  if (prim->IsInstance<TypeName##Node>()) {                                       \
    const auto& binary = Downcast<TypeName>(prim);                                \
    return MatchOrCreatePrim(prim, "", {AddPrim(binary->a), AddPrim(binary->b)}); \
  }

const MSCPrim RelaxGraphBuilder::AddPrim(const PrimExpr& prim) {
  if (prim_map_.count(prim)) {
    return prim_map_[prim];
  }

  // binary
  ADD_BINARY_PRIM(tvm::tir::Add)
  ADD_BINARY_PRIM(tvm::tir::Sub)
  ADD_BINARY_PRIM(tvm::tir::Mul)
  ADD_BINARY_PRIM(tvm::tir::Div)
  ADD_BINARY_PRIM(tvm::tir::Mod)
  ADD_BINARY_PRIM(tvm::tir::FloorDiv)
  ADD_BINARY_PRIM(tvm::tir::FloorMod)
  ADD_BINARY_PRIM(tvm::tir::Max)
  ADD_BINARY_PRIM(tvm::tir::Min)

  // compare
  ADD_BINARY_PRIM(tvm::tir::EQ)
  ADD_BINARY_PRIM(tvm::tir::NE)
  ADD_BINARY_PRIM(tvm::tir::LT)
  ADD_BINARY_PRIM(tvm::tir::LE)
  ADD_BINARY_PRIM(tvm::tir::GT)
  ADD_BINARY_PRIM(tvm::tir::GE)

  // scalar
  if (prim->IsInstance<IntImmNode>()) {
    Map<String, String> attrs;
    attrs.Set("value", StringUtils::ToString(prim));
    return MatchOrCreatePrim(prim, "Int", Array<BaseJoint>(), attrs);
  }

  // call
  if (const auto* c_node = prim.as<tvm::tir::CallNode>()) {
    String optype;
    Array<BaseJoint> parents;
    if (const auto* op_node = c_node->op.as<OpNode>()) {
      optype = StringUtils::Replace(op_node->name, "tir.", "");
    } else {
      optype = "Prim";
    }
    for (const auto& a : c_node->args) {
      parents.push_back(AddPrim(a));
    }
    return MatchOrCreatePrim(prim, optype, parents);
  }
  return MatchOrCreatePrim(prim);
}

const MSCPrim RelaxGraphBuilder::MatchOrCreatePrim(const PrimExpr& prim, const String& optype,
                                                   const Array<BaseJoint>& parents,
                                                   const Map<String, String>& attrs) {
  if (prim_map_.count(prim)) {
    return prim_map_[prim];
  }
  const auto& op_ =
      optype.size() == 0 ? StringUtils::Replace(prim->GetTypeKey(), "tir.", "") : optype;
  for (const auto& p : prims_) {
    if (p->optype != op_ || p->attrs.size() != attrs.size() ||
        p->parents.size() != parents.size()) {
      continue;
    }
    bool attrs_match = std::all_of(p->attrs.begin(), p->attrs.end(), [&attrs](const auto& pair) {
      return attrs.count(pair.first) && attrs[pair.first] == pair.second;
    });
    if (!attrs_match) {
      continue;
    }
    bool parents_match = true;
    for (size_t i = 0; i < parents.size(); i++) {
      if (p->ParentAt(i)->name != parents[i]->name) {
        parents_match = false;
        break;
      }
    }
    if (!parents_match) {
      continue;
    }
    prim_map_.Set(prim, p);
    return p;
  }
  String name;
  if (const auto* v_node = prim.as<tvm::tir::VarNode>()) {
    name = v_node->name_hint;
  } else {
    name = StringUtils::Upper(op_) + "_" + std::to_string(prims_.size());
  }
  const auto& node = MSCPrim(prims_.size(), name, op_, parents, attrs);
  prims_.push_back(node);
  prim_map_.Set(prim, node);
  return node;
}

void RelaxGraphBuilder::VisitExpr_(const relax::ConstantNode* op) {
  if (!expr_tensor_map_.count(GetRef<relax::Constant>(op))) {
    AddNode(GetRef<relax::Constant>(op));
  }
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::ConstantNode* val) {
  const String& name = config_.use_var_name ? binding->var->name_hint() : "";
  AddNode(GetRef<relax::Constant>(val), binding->var, name);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::ShapeExprNode* val) {
  const String& name = config_.use_var_name ? binding->var->name_hint() : "";
  AddNode(GetRef<relax::ShapeExpr>(val), binding->var, name);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::CallNode* call_node) {
  RelaxExprVisitor::VisitBinding_(binding, call_node);
  const String& name = config_.use_var_name ? binding->var->name_hint() : "";
  try {
    AddNode(GetRef<relax::Call>(call_node), binding->var, name);
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to add node from " << binding->var << " : " << binding->value
                 << ", reason: " << err.message();
    throw err;
  }
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::TupleNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  const String& name = config_.use_var_name ? binding->var->name_hint() : "";
  AddNode(GetRef<relax::Tuple>(val), binding->var, name);
}

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::TupleGetItemNode* val) {
  RelaxExprVisitor::VisitBinding_(binding, val);
  const String& name = config_.use_var_name ? binding->var->name_hint() : "";
  AddNode(GetRef<relax::TupleGetItem>(val), binding->var, name);
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

void RelaxGraphBuilder::VisitBinding_(const relax::VarBindingNode* binding,
                                      const relax::FunctionNode* val) {
  const auto& name_opt = val->GetAttr<runtime::String>(relay::attr::kComposite);
  ICHECK(name_opt.defined()) << "Unexpected target func without composite";
  ICHECK(config_.target.size() > 0 && StringUtils::StartsWith(name_opt.value(), config_.target))
      << "Target should be given for target function";
  target_funcs_.Set(binding->var, GetRef<relax::Function>(val));
}

const std::tuple<String, String, String> RelaxGraphBuilder::ParseFunc(const relax::Function& func) {
  String node_name, optype, layout;
  const auto& name_opt = func->GetAttr<runtime::String>(msc_attr::kUnique);
  // get node_name
  if (name_opt.defined()) {
    node_name = name_opt.value();
  }
  // get optype
  const auto& codegen_opt = func->GetAttr<runtime::String>(relax::attr::kCodegen);
  const auto& optype_opt = func->GetAttr<runtime::String>(msc_attr::kOptype);
  const auto& composite_opt = func->GetAttr<runtime::String>(relax::attr::kComposite);
  if (codegen_opt.defined()) {
    optype = codegen_opt.value();
  } else if (optype_opt.defined()) {
    optype = optype_opt.value();
  } else if (composite_opt.defined()) {
    optype = composite_opt.value();
    if (config_.target.size() > 0) {
      optype = StringUtils::Replace(composite_opt.value(), config_.target + ".", "");
    }
  }
  // get layout
  const auto& layout_opt = func->GetAttr<runtime::String>(msc_attr::kLayout);
  if (layout_opt.defined()) {
    layout = layout_opt.value();
  }
  return std::make_tuple(node_name, optype, layout);
}

void RelaxGraphBuilder::VisitPrimExpr(const PrimExpr& prim) {
  RelaxExprVisitor::VisitPrimExpr(prim);
  if (!prim->IsInstance<IntImmNode>() && !prim->IsInstance<FloatImmNode>()) {
    AddPrim(prim);
  }
}

Array<Expr> RelaxGraphBuilder::GetPluginInputs(const relax::Expr& expr) {
  ICHECK(expr->IsInstance<relax::CallNode>()) << "plugin expr should be call";
  const auto& call = Downcast<relax::Call>(expr);
  ICHECK(call->args[1]->IsInstance<relax::TupleNode>()) << "plugin argument 1 should be call";
  return Downcast<relax::Tuple>(call->args[1])->fields;
}

Map<MSCTensor, NDArray> RelaxWeightsExtractor::GetWeights(const relax::Function& func) {
  VisitExpr(func);
  return weights_;
}

void RelaxWeightsExtractor::VisitExpr_(const relax::ConstantNode* op) {
  const auto& name = SpanUtils::GetAttr(op->span, msc_attr::kName);
  const auto& layout = SpanUtils::GetAttr(op->span, msc_attr::kLayout);
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

void RelaxWeightsExtractor::VisitExpr_(const relax::CallNode* op) {
  RelaxExprVisitor::VisitExpr_(op);
  if (const auto* v_node = op->op.as<GlobalVarNode>()) {
    const auto& func = Downcast<relax::Function>(ref_module_->Lookup(v_node->name_hint));
    VisitExpr(func);
  }
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
  const auto& node_name = name.size() > 0 ? name : SpanUtils::GetAttr(expr->span, msc_attr::kName);
  const auto& shared_ref = SpanUtils::GetAttr(expr->span, msc_attr::kSharedRef);

  // Get optype
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
      attrs.Set("scalar", GetScalarStr(const_node->data, config_.float_precision));
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
        const auto& weight_name = SpanUtils::GetAttr(arg->span, msc_attr::kName);
        const auto& pair = tensor_input_map_[t_name];
        const auto& producer = Downcast<MSCJoint>(pair.first);
        if (!weights_.count(weight_name)) {
          const auto& ref = producer->OutputAt(pair.second);
          MSCTensor weight;
          if (input_types[i] == "bias") {
            weight = MSCTensor(weight_name, ref->dtype, "O", Array<Integer>{ref->GetSize()});
          } else {
            weight = MSCTensor(weight_name, ref->dtype, ref->layout.name(), ref->shape);
          }
          weights_.Set(weight_name, weight);
        }
        if (producer->HasAttr("scalar")) {
          attrs.Set(input_types[i], producer->GetTypeAttr<std::string>("scalar"));
        }
        node_weights.Set(input_types[i], weights_[weight_name]);
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
  } else if (optype == "constant") {
    Type checked_type = expr->checked_type_;
    ICHECK(checked_type.defined() && checked_type->IsInstance<relay::TensorTypeNode>())
        << "Constant checked_type is not defined";
    const auto& t_info = Downcast<TensorType>(checked_type);
    const auto& layout = SpanUtils::GetAttr(expr->span, msc_attr::kLayout);
    const auto& weight =
        MSCTensor(node_name, t_info->dtype, layout, ArrayUtils::Cast<Integer>(t_info->shape));
    node_weights.Set("const", weight);
  }
  std::vector<std::pair<BaseJoint, size_t>> inputs;
  for (const auto& i : input_names) {
    inputs.push_back(tensor_input_map_[i]);
  }

  // Build outputs
  Array<MSCTensor> outputs;
  const auto& layout = SpanUtils::GetAttr(expr->span, msc_attr::kLayout);
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
    StartFuncScope(SpanUtils::GetAttr(op->span, msc_attr::kName));
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
        if (!expr_tensor_map_.count(op->args[i])) {
          RelayExprVisitor::VisitExpr(op->args[i]);
        }
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
  const auto& name = SpanUtils::GetAttr(op->span, msc_attr::kName);
  const auto& layout = SpanUtils::GetAttr(op->span, msc_attr::kLayout);
  const auto& t_info = op->tensor_type();
  const auto& shape = ArrayUtils::Cast<Integer>(t_info->shape);
  const auto& weight = MSCTensor(name, t_info->dtype, layout, shape);
  weights_.Set(weight, op->data);
}

TVM_REGISTER_GLOBAL("msc.core.BuildFromRelax")
    .set_body_typed([](const IRModule& relax_module, const String& entry_name,
                       const String& options) -> MSCGraph {
      auto builder = RelaxGraphBuilder(relax_module, entry_name, options);
      const auto& func_name =
          builder.config().byoc_entry.size() > 0 ? String(builder.config().byoc_entry) : entry_name;
      const auto& func = Downcast<relax::Function>(relax_module->Lookup(func_name));
      return builder.Build(func);
    });

TVM_REGISTER_GLOBAL("msc.core.GetRelaxWeights")
    .set_body_typed([](const IRModule& relax_module,
                       const String& entry_name) -> Map<MSCTensor, NDArray> {
      const auto& func = Downcast<relax::Function>(relax_module->Lookup(entry_name));
      return RelaxWeightsExtractor(relax_module).GetWeights(func);
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
