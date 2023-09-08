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
 * \file src/contrib/msc/core/transform/set_expr_name.cc
 * \brief Pass for setting name for call and constant.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../utils.h"

namespace tvm {
using namespace tvm::contrib::msc;

namespace relax {

/*!
 * \brief Name setter for Relax
 */
class RelaxExprNameSetter : public ExprVisitor {
 public:
  explicit RelaxExprNameSetter(const IRModule& ref_module) : ref_module_(ref_module) {}

  void VisitBindingBlock(const BindingBlock& block) final {
    String block_name = SpanUtils::GetAttr(block->span, "name");
    if (block_name.size() == 0) {
      block_name = "block";
    }
    if (setted_blocks_.count(block_name)) {
      int cnt = 1;
      while (setted_blocks_.count(block_name + "_" + std::to_string(cnt))) {
        cnt++;
      }
      block_name = block_name + "_" + std::to_string(cnt);
    }
    setted_blocks_.insert(block_name);
    block_stack_.push_back(block_name);
    const String& unique_name = StringUtils::Join(block_stack_, ".");
    block->span = SpanUtils::SetAttr(block->span, "name", unique_name);
    ExprVisitor::VisitBindingBlock(block);
    block_stack_.pop_back();
  }

  void VisitExpr_(const ConstantNode* val) {
    ExprVisitor::VisitExpr_(val);
    const String& unique_name = GetUniqueName(GetRef<Constant>(val), "const");
    if (unique_name != SpanUtils::GetAttr(val->span, "name")) {
      val->span = SpanUtils::SetAttr(val->span, "name", unique_name);
    }
    expr_names_.Set(GetRef<Constant>(val), unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    const String& unique_name = GetUniqueName(GetRef<Constant>(val), "const");
    if (unique_name != SpanUtils::GetAttr(val->span, "name")) {
      val->span = SpanUtils::SetAttr(val->span, "name", unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const ShapeExprNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    const String& unique_name = GetUniqueName(GetRef<ShapeExpr>(val), "shape");
    if (unique_name != SpanUtils::GetAttr(val->span, "name")) {
      val->span = SpanUtils::SetAttr(val->span, "name", unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    const String& unique_name = GetUniqueName(GetRef<Tuple>(val), "tuple");
    if (unique_name != SpanUtils::GetAttr(val->span, "name")) {
      val->span = SpanUtils::SetAttr(val->span, "name", unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    ICHECK(expr_names_.count(val->tuple)) << "Can not find tuple of " << GetRef<TupleGetItem>(val);
    const String& unique_name = expr_names_[val->tuple] + "." + std::to_string(val->index);
    if (unique_name != SpanUtils::GetAttr(val->span, "name")) {
      val->span = SpanUtils::SetAttr(val->span, "name", unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    String name_hint, optype;
    if (const auto* op_node = val->op.as<OpNode>()) {
      const std::string& op_name = op_node->name;
      int rpos = op_name.rfind(".");
      name_hint = op_name.substr(rpos + 1);
      optype = StringUtils::Replace(op_node->name, "relax.", "");
    } else if (const auto* v_node = val->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<Function>(ref_module_->Lookup(v_node->name_hint));
      ExprVisitor::VisitExpr(func);
      const auto& name_opt = func->GetAttr<runtime::String>(attr::kComposite);
      ICHECK(name_opt.defined()) << "Unexpected global func without composite";
      name_hint = name_opt.value();
      optype = name_hint;
    }
    // set name
    const String& unique_name = GetUniqueName(GetRef<Expr>(val), name_hint);
    if (unique_name != SpanUtils::GetAttr(val->span, "name")) {
      val->span = SpanUtils::SetAttr(val->span, "name", unique_name);
    }
    // set constant consumer && shared_ref
    Array<String> input_types;
    try {
      input_types = ExprUtils::GetInputTypes(optype, val->args.size(), true);
    } catch (runtime::InternalError& err) {
      LOG(WARNING) << "Failed to GetInputTypes for " << GetRef<Call>(val) << " : " << err.message();
      throw err;
    }
    for (size_t i = 0; i < input_types.size(); i++) {
      if (input_types[i] == "input") {
        continue;
      }
      if (const auto* c_node = val->args[i].as<ConstantNode>()) {
        const String& const_name = SpanUtils::GetAttr(c_node->span, "name");
        if (constant_consumers_.count(const_name)) {
          val->span = SpanUtils::SetAttr(val->span, "shared_ref", constant_consumers_[const_name]);
        } else {
          constant_consumers_.Set(const_name, unique_name);
        }
      }
    }
    expr_names_.Set(binding->var, unique_name);
  }

 private:
  const String GetUniqueName(const Expr& expr, const String& name_hint) {
    String expr_name = SpanUtils::GetAttr(expr->span, "name");
    if (expr_name.size() == 0) {
      expr_name = name_hint;
    }
    if (!setted_names_.count(expr_name)) {
      setted_names_.Set(expr_name, expr);
      return expr_name;
    }
    if (setted_names_[expr_name] == expr) {
      return expr_name;
    }
    int cnt = 1;
    while (setted_names_.count(expr_name + "_" + std::to_string(cnt)) &&
           setted_names_[expr_name + "_" + std::to_string(cnt)] != expr) {
      cnt++;
    }
    expr_name = expr_name + "_" + std::to_string(cnt);
    if (!setted_names_.count(expr_name)) {
      setted_names_.Set(expr_name, expr);
    }
    return expr_name;
  }

  Map<String, Expr> setted_names_;
  Map<String, String> constant_consumers_;
  std::set<String> setted_blocks_;
  Array<String> block_stack_;
  Map<Expr, String> expr_names_;
  IRModule ref_module_;
};  // class ExprNameSetter

void SetRelaxExprName(const IRModule& ref_module, const Expr& e) {
  RelaxExprNameSetter(ref_module).VisitExpr(e);
}

namespace transform {

Pass SetRelaxExprName(const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    relax::SetRelaxExprName(m, m->Lookup(entry_name));
    return m;
  };
  return CreateModulePass(pass_func, 0, "SetRelaxExprName", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SetRelaxExprName").set_body_typed(SetRelaxExprName);

}  // namespace transform
}  // namespace relax

namespace relay {

/*!
 * \brief Name setter for Relay
 */
class RelayExprNameSetter : public ExprVisitor {
 public:
  explicit RelayExprNameSetter(const IRModule& ref_module) : ref_module_(ref_module) {}

  void VisitExpr_(const ConstantNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& unique_name = GetUniqueName(GetRef<Constant>(op), "const");
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
  }

  void VisitExpr_(const TupleNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& unique_name = GetUniqueName(GetRef<Tuple>(op), "tuple");
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& tuple_name = SpanUtils::GetAttr(op->tuple->span, "name");
    const String& unique_name = tuple_name + "." + std::to_string(op->index);
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
  }

  void VisitExpr_(const FunctionNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const auto& name_opt = op->GetAttr<runtime::String>(attr::kComposite);
    const String& name_hint = name_opt.defined() ? name_opt.value() : "func";
    const String& unique_name = GetUniqueName(GetRef<Function>(op), name_hint);
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    ExprVisitor::VisitExpr_(op);
    String name_hint, optype;
    if (const auto* op_node = op->op.as<OpNode>()) {
      const std::string& op_name = op_node->name;
      int rpos = op_name.rfind(".");
      name_hint = op_name.substr(rpos + 1);
      optype = StringUtils::Replace(op_node->name, "relay.", "");
    } else if (const auto* v_node = op->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<Function>(ref_module_->Lookup(v_node->name_hint));
      ExprVisitor::VisitExpr(func);
      const auto& name_opt = func->GetAttr<runtime::String>(attr::kComposite);
      ICHECK(name_opt.defined()) << "Unexpected global func without composite";
      optype = name_opt.value();
      name_hint = optype;
    }
    // set name
    const String& unique_name = GetUniqueName(GetRef<Expr>(op), name_hint);
    if (unique_name != SpanUtils::GetAttr(op->span, "name")) {
      op->span = SpanUtils::SetAttr(op->span, "name", unique_name);
    }
    // set constant consumer && shared_ref
    Array<String> input_types;
    try {
      input_types = ExprUtils::GetInputTypes(optype, op->args.size(), false);
    } catch (runtime::InternalError& err) {
      LOG(WARNING) << "Failed to GetInputTypes for " << GetRef<Call>(op) << " : " << err.message();
      throw err;
    }
    for (size_t i = 0; i < input_types.size(); i++) {
      if (input_types[i] == "input") {
        continue;
      }
      if (const auto* c_node = op->args[i].as<ConstantNode>()) {
        const String& const_name = SpanUtils::GetAttr(c_node->span, "name");
        if (constant_consumers_.count(const_name)) {
          op->span = SpanUtils::SetAttr(op->span, "shared_ref", constant_consumers_[const_name]);
        } else {
          constant_consumers_.Set(const_name, unique_name);
        }
      }
    }
  }

 private:
  const String GetUniqueName(const Expr& expr, const String& name_hint) {
    String expr_name = SpanUtils::GetAttr(expr->span, "name");
    if (expr_name.size() == 0) {
      expr_name = name_hint;
    }
    if (!setted_names_.count(expr_name)) {
      setted_names_.Set(expr_name, expr);
      return expr_name;
    }
    if (setted_names_[expr_name] == expr) {
      return expr_name;
    }
    int cnt = 1;
    while (setted_names_.count(expr_name + "_" + std::to_string(cnt)) &&
           setted_names_[expr_name + "_" + std::to_string(cnt)] != expr) {
      cnt++;
    }
    expr_name = expr_name + "_" + std::to_string(cnt);
    if (!setted_names_.count(expr_name)) {
      setted_names_.Set(expr_name, expr);
    }
    return expr_name;
  }

  Map<String, Expr> setted_names_;
  Map<String, String> constant_consumers_;
  IRModule ref_module_;
};  // class ExprNameSetter

void SetRelayExprName(const IRModule& ref_module, const Expr& e) {
  RelayExprNameSetter(ref_module).VisitExpr(e);
}

namespace transform {

Pass SetRelayExprName(const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    relay::SetRelayExprName(m, m->Lookup(entry_name));
    return m;
  };
  return CreateModulePass(pass_func, 0, "SetRelayExprName", {});
}

TVM_REGISTER_GLOBAL("relay._transform.SetRelayExprName").set_body_typed(SetRelayExprName);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
