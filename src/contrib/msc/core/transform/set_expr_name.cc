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

class FuncNameGetter : public ExprVisitor {
 public:
  explicit FuncNameGetter(const Array<String>& arg_names) : arg_names_(arg_names) {}

  /*! \brief Get the attributes from prim value as Map<String, String>*/
  String HintName(const Expr& expr) {
    name_ = "";
    ExprVisitor::VisitExpr(expr);
    return name_;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) {
    if (name_.size() == 0) {
      name_ = SpanUtils::GetAttr(val->span, msc_attr::kName);
    }
    if (name_.size() == 0) {
      ExprVisitor::VisitBinding_(binding, val);
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) {
    if (name_.size() == 0) {
      name_ = SpanUtils::GetAttr(val->span, msc_attr::kName);
    }
    if (name_.size() == 0) {
      ExprVisitor::VisitBinding_(binding, val);
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) {
    if (name_.size() == 0 && arg_names_[0].size() > 0) {
      name_ = arg_names_[0] + "." + std::to_string(val->index);
    }
    if (name_.size() == 0) {
      ExprVisitor::VisitBinding_(binding, val);
    }
  }

 private:
  String name_;
  Array<String> arg_names_;
};

/*!
 * \brief Name setter for Relax
 */
class RelaxExprNameSetter : public ExprVisitor {
 public:
  explicit RelaxExprNameSetter(const IRModule& ref_module, const String& target,
                               const Map<String, String>& var_names)
      : ref_module_(ref_module), target_{target}, var_names_{var_names} {}

  void VisitBindingBlock(const BindingBlock& block) final {
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
    setted_blocks_.insert(prefix + "." + block_name);
    block_stack_.push_back(block_name);
    const String& unique_name = StringUtils::Join(block_stack_, ".");
    block->span = SpanUtils::SetAttr(block->span, msc_attr::kName, unique_name);
    ExprVisitor::VisitBindingBlock(block);
    block_stack_.pop_back();
  }

  void VisitExpr_(const ConstantNode* val) {
    ExprVisitor::VisitExpr_(val);
    const String& unique_name = GetUniqueName(GetRef<Constant>(val), "const");
    if (unique_name != SpanUtils::GetAttr(val->span, msc_attr::kName)) {
      val->span = SpanUtils::SetAttr(val->span, msc_attr::kName, unique_name);
    }
    expr_names_.Set(GetRef<Constant>(val), unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    const String& unique_name = GetUniqueName(GetRef<Constant>(val), "const");
    if (unique_name != SpanUtils::GetAttr(val->span, msc_attr::kName)) {
      val->span = SpanUtils::SetAttr(val->span, msc_attr::kName, unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const ShapeExprNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    const String& unique_name = GetUniqueName(GetRef<ShapeExpr>(val), "shape");
    if (unique_name != SpanUtils::GetAttr(val->span, msc_attr::kName)) {
      val->span = SpanUtils::SetAttr(val->span, msc_attr::kName, unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    const String& unique_name = GetUniqueName(GetRef<Tuple>(val), "tuple");
    if (unique_name != SpanUtils::GetAttr(val->span, msc_attr::kName)) {
      val->span = SpanUtils::SetAttr(val->span, msc_attr::kName, unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    String unique_name;
    if (expr_names_.count(val->tuple)) {
      unique_name = expr_names_[val->tuple] + "." + std::to_string(val->index);
    } else if (const auto* v_node = val->tuple.as<VarNode>()) {
      unique_name = v_node->name_hint() + "." + std::to_string(val->index);
    }
    if (unique_name != SpanUtils::GetAttr(val->span, msc_attr::kName)) {
      val->span = SpanUtils::SetAttr(val->span, msc_attr::kName, unique_name);
    }
    expr_names_.Set(binding->var, unique_name);
  }

  void VisitBinding_(const VarBindingNode* binding, const FunctionNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    const auto& name_opt = val->GetAttr<runtime::String>(attr::kComposite);
    if (name_opt.defined()) {
      local_funcs_.Set(binding->var, GetRef<Function>(val));
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) {
    ExprVisitor::VisitBinding_(binding, val);
    String name_hint, optype;
    bool use_unique = true;
    if (var_names_.count(binding->var->name_hint())) {
      name_hint = var_names_[binding->var->name_hint()];
    } else if (const auto* op_node = val->op.as<OpNode>()) {
      const std::string& op_name = op_node->name;
      if (op_name == "relax.call_dps_packed" && val->args[0]->IsInstance<ExternFuncNode>()) {
        const auto& func = Downcast<ExternFunc>(val->args[0]);
        name_hint = func->global_symbol;
        optype = func->global_symbol;
        const String& input_name = GetUniqueName(val->args[1], "plugin_inputs");
        if (input_name != SpanUtils::GetAttr(val->args[1]->span, msc_attr::kName)) {
          val->args[1]->span = SpanUtils::SetAttr(val->args[1]->span, msc_attr::kName, input_name);
        }
      } else {
        int rpos = op_name.rfind(".");
        name_hint = op_name.substr(rpos + 1);
        optype = StringUtils::Replace(op_node->name, "relax.", "");
      }
    } else if (const auto* v_node = val->op.as<GlobalVarNode>()) {
      const auto& func = Downcast<Function>(ref_module_->Lookup(v_node->name_hint));
      ExprVisitor::VisitExpr(func);
      optype = GetFuncType(func);
      name_hint = GetFuncName(GetRef<Call>(val), func);
      use_unique = false;
    } else if (local_funcs_.count(val->op)) {
      ExprVisitor::VisitExpr(local_funcs_[val->op]);
      optype = GetFuncType(local_funcs_[val->op]);
      name_hint = GetFuncName(GetRef<Call>(val), local_funcs_[val->op]);
      use_unique = false;
    }
    if (name_hint.size() > 0) {
      // set name
      const String& unique_name =
          use_unique ? GetUniqueName(GetRef<Expr>(val), name_hint) : name_hint;
      if (unique_name != SpanUtils::GetAttr(val->span, msc_attr::kName)) {
        val->span = SpanUtils::SetAttr(val->span, msc_attr::kName, unique_name);
      }
      // set constant consumer && shared_ref
      Array<String> input_types;
      try {
        input_types = ExprUtils::GetInputTypes(optype, val->args.size(), true);
      } catch (runtime::InternalError& err) {
        LOG(WARNING) << "Failed to GetInputTypes for " << GetRef<Call>(val) << " : "
                     << err.message();
        throw err;
      }
      for (size_t i = 0; i < input_types.size(); i++) {
        if (input_types[i] == "input") {
          continue;
        }
        if (const auto* c_node = val->args[i].as<ConstantNode>()) {
          const String& const_name = SpanUtils::GetAttr(c_node->span, msc_attr::kName);
          if (constant_consumers_.count(const_name)) {
            val->span = SpanUtils::SetAttr(val->span, msc_attr::kSharedRef,
                                           constant_consumers_[const_name]);
          } else {
            constant_consumers_.Set(const_name, unique_name);
          }
        }
      }
      expr_names_.Set(binding->var, unique_name);
    }
  }

 private:
  const String GetUniqueName(const Expr& expr, const String& name_hint) {
    String expr_name = SpanUtils::GetAttr(expr->span, msc_attr::kName);
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

  const String GetFuncType(const Function& func) {
    String optype;
    const auto& comp_opt = func->GetAttr<runtime::String>(attr::kComposite);
    const auto& code_opt = func->GetAttr<runtime::String>(attr::kCodegen);
    if (comp_opt.defined()) {
      optype = comp_opt.value();
    } else if (code_opt.defined()) {
      optype = code_opt.value();
    } else {
      optype = "extern_func";
    }
    if (target_.size() > 0) {
      optype = StringUtils::Replace(optype, target_ + ".", "");
    }
    return optype;
  }

  const String GetFuncName(const Call& call, const Function& func) {
    String name;
    // get from unique
    const auto& name_opt = func->GetAttr<runtime::String>(msc_attr::kUnique);
    if (name_opt.defined()) {
      return name_opt.value();
    }
    // get from exprs in the func
    Array<String> arg_names;
    for (const auto& a : call->args) {
      arg_names.push_back(expr_names_.count(a) ? expr_names_[a] : "");
    }
    name = FuncNameGetter(arg_names).HintName(local_funcs_[call->op]);
    if (name.size() > 0) {
      return name;
    }
    const auto& optype = GetFuncType(func);
    if (optype == "extern_func") {
      name = Downcast<Var>(call->op)->name_hint();
    } else {
      name = optype;
    }
    return GetUniqueName(call, name);
  }

  Map<String, Expr> setted_names_;
  Map<String, String> constant_consumers_;
  std::set<String> setted_blocks_;
  Array<String> block_stack_;
  Map<Expr, String> expr_names_;
  Map<Expr, Function> local_funcs_;
  IRModule ref_module_;
  String target_;
  Map<String, String> var_names_;
};  // class ExprNameSetter

void SetRelaxExprName(const IRModule& ref_module, const Expr& e, const String& target,
                      const Map<String, String>& var_names) {
  RelaxExprNameSetter(ref_module, target, var_names).VisitExpr(e);
}

namespace transform {

Pass SetRelaxExprName(const String& entry_name, const String& target,
                      const Map<String, String>& var_names) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    relax::SetRelaxExprName(m, m->Lookup(entry_name), target, var_names);
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
    if (unique_name != SpanUtils::GetAttr(op->span, msc_attr::kName)) {
      op->span = SpanUtils::SetAttr(op->span, msc_attr::kName, unique_name);
    }
  }

  void VisitExpr_(const TupleNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& unique_name = GetUniqueName(GetRef<Tuple>(op), "tuple");
    if (unique_name != SpanUtils::GetAttr(op->span, msc_attr::kName)) {
      op->span = SpanUtils::SetAttr(op->span, msc_attr::kName, unique_name);
    }
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const String& tuple_name = SpanUtils::GetAttr(op->tuple->span, msc_attr::kName);
    const String& unique_name = tuple_name + "." + std::to_string(op->index);
    if (unique_name != SpanUtils::GetAttr(op->span, msc_attr::kName)) {
      op->span = SpanUtils::SetAttr(op->span, msc_attr::kName, unique_name);
    }
  }

  void VisitExpr_(const FunctionNode* op) final {
    ExprVisitor::VisitExpr_(op);
    const auto& name_opt = op->GetAttr<runtime::String>(attr::kComposite);
    const String& name_hint = name_opt.defined() ? name_opt.value() : "func";
    const String& unique_name = GetUniqueName(GetRef<Function>(op), name_hint);
    if (unique_name != SpanUtils::GetAttr(op->span, msc_attr::kName)) {
      op->span = SpanUtils::SetAttr(op->span, msc_attr::kName, unique_name);
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
      const auto& name_opt = func->GetAttr<runtime::String>(attr::kComposite);
      if (name_opt.defined()) {
        optype = name_opt.value();
        name_hint = optype;
        ExprVisitor::VisitExpr(func);
      } else {
        optype = "extern_func";
        name_hint = v_node->name_hint;
      }
    }
    if (name_hint.size() > 0) {
      // set name
      const String& unique_name = GetUniqueName(GetRef<Expr>(op), name_hint);
      if (unique_name != SpanUtils::GetAttr(op->span, msc_attr::kName)) {
        op->span = SpanUtils::SetAttr(op->span, msc_attr::kName, unique_name);
      }
      // set constant consumer && shared_ref
      Array<String> input_types;
      try {
        input_types = ExprUtils::GetInputTypes(optype, op->args.size(), false);
      } catch (runtime::InternalError& err) {
        LOG(WARNING) << "Failed to GetInputTypes for " << GetRef<Call>(op) << " : "
                     << err.message();
        throw err;
      }
      for (size_t i = 0; i < input_types.size(); i++) {
        if (input_types[i] == "input") {
          continue;
        }
        if (const auto* c_node = op->args[i].as<ConstantNode>()) {
          const String& const_name = SpanUtils::GetAttr(c_node->span, msc_attr::kName);
          if (constant_consumers_.count(const_name)) {
            op->span =
                SpanUtils::SetAttr(op->span, msc_attr::kSharedRef, constant_consumers_[const_name]);
          } else {
            constant_consumers_.Set(const_name, unique_name);
          }
        }
      }
    }
  }

 private:
  const String GetUniqueName(const Expr& expr, const String& name_hint) {
    String expr_name = SpanUtils::GetAttr(expr->span, msc_attr::kName);
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

/*!
 * \brief Name binder for Relay
 */
class RelayExprNameBinder : public ExprVisitor {
 public:
  explicit RelayExprNameBinder(const String& name_key, const String& seperator)
      : name_key_(name_key), seperator_(seperator) {}

  void VisitExpr_(const ConstantNode* op) final {
    if (op->span.defined()) {
      BindName(GetRef<Constant>(op));
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->span.defined()) {
      BindName(GetRef<Call>(op));
    }
    ExprVisitor::VisitExpr_(op);
  }

 private:
  void BindName(const Expr& expr) {
    const auto& name = expr->span->source_name->name;
    String valid_name;
    if (name_key_.size() == 0) {
      valid_name = name;
      expr->span = Span(SourceName::Get(""), expr->span->line, expr->span->end_line,
                        expr->span->column, expr->span->end_column);
    } else {
      String right = std::get<1>(StringUtils::SplitOnce(name, name_key_));
      if (right.size() > 0) {
        valid_name = std::get<0>(StringUtils::SplitOnce(name, seperator_));
        if (valid_name.size() > 0) {
          const auto& new_source = StringUtils::Replace(name, name_key_ + valid_name, "");
          expr->span = Span(SourceName::Get(new_source), expr->span->line, expr->span->end_line,
                            expr->span->column, expr->span->end_column);
        }
      }
    }
    if (valid_name.size() > 0) {
      if (setted_names_.count(valid_name)) {
        int cnt = 1;
        while (setted_names_.count(valid_name + "_" + std::to_string(cnt)) &&
               setted_names_[valid_name + "_" + std::to_string(cnt)] != expr) {
          cnt++;
        }
        valid_name = valid_name + "_" + std::to_string(cnt);
      }
      setted_names_.Set(valid_name, expr);
      expr->span = SpanUtils::SetAttr(expr->span, msc_attr::kName, valid_name);
    }
  }

  Map<String, Expr> setted_names_;
  String name_key_;
  String seperator_;
};  // class ExprNameBinder

void BindRelayExprName(const Expr& e, const String& name_key, const String& seperator) {
  RelayExprNameBinder(name_key, seperator).VisitExpr(e);
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

Pass BindRelayExprName(const String& name_key, const String& seperator, const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    relay::BindRelayExprName(m->Lookup(entry_name), name_key, seperator);
    return m;
  };
  return CreateModulePass(pass_func, 0, "BindRelayExprName", {});
}

TVM_REGISTER_GLOBAL("relay._transform.BindRelayExprName").set_body_typed(BindRelayExprName);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
