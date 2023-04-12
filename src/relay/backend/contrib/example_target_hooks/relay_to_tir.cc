
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
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/memory.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include "../../../op/call/call.h"
#include "tvm/tir/function.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace example_target_hooks {

namespace {

/*!
 * \brief An example mutator for a "RelayToTIR" custom pass. Replaces every call to a Relay
 * Function with "external_symbol" attribute of "replace_add_with_subtract" with a call to a
 * TIR PrimFunc implementing subtraction.
 *
 * Illustrates six aspects a custom 'lowering' style pass may need to account for:
 *  - Lowerable functions can appear inline as call ops, bound to let-bound variables, or as
 *    global functions.
 *  - Let-bound lowerable functions should be inlined on-the-fly since after processing the
 *    let-binding is no longer required.
 *  - There may be multiple calls to the same lowerable function. All calls need to be
 *    rewritten, even though the function itself need be rewritten only once.
 *  - GlobalVars must be shared between all calls and the new definition itself.
 *  - Calls to lowered functions must use the "call_lowered" calling convention.
 *  - The Target::Current() may hold an instance of the TargetKind from which the custom Pass
 *    was extracted.
 *
 * Though not illustrated here, it is also valid for a "RelayToTIR" custom pass to add
 * runtime::Modules to the output IRModule's "external_mods" attribute. In this case the
 * IRModule must be left with an 'extern' Function definition with the matching "external_symbol"
 * name.
 */
class ConvertAddToSubtract : public MixedModeMutator {
 public:
  explicit ConvertAddToSubtract(IRModule ir_module, Target host_target)
      : ir_module_(ir_module),
        host_target_(host_target),
        custom_target_(Target("example_target_hook")) {}

  IRModule Mutate() {
    GlobalVar main_global_var = ir_module_->GetGlobalVar("main");
    Function main = Downcast<Function>(ir_module_->Lookup(main_global_var));
    Function mutated_main = WithFields(main, main->params, VisitExpr(main->body));

    ir_module_->Update(main_global_var, mutated_main);

    return ir_module_;
  }

 private:
  tir::BufferLoad LoadIndex(const tir::Buffer& buffer, const PrimExpr& index) {
    return tir::BufferLoad(buffer, {index});
  }

  GlobalVar ReplaceAddWithSubtractPrimFunc(const Function& func) {
    auto func_name = func->GetAttr<String>(::tvm::attr::kGlobalSymbol);
    ICHECK(func_name.defined());

    // --------------------------------------------------------------------------------------------
    // Cases:
    //  - Inline function:
    //     - First encounter: create global var, rewrite to PrimFunc, add binding, replace call.
    //     - Thereafter (via object sharing): discover global var already in module, replace call
    //  - Global function:
    //     - Assume func_name == global_var->name_hint
    //     - First encounter: create global var, rewrite to PrimFunc, update binding, replace call
    //     - Thereafter (via global var): discover global var already in module, replace call
    // --------------------------------------------------------------------------------------------

    // If necessary, introduce a new global var to map the function to and copy the source type
    // over for InferType.
    GlobalVar global_var;
    bool need_rewriting;
    if (ir_module_->ContainGlobalVar(func_name.value())) {
      global_var = ir_module_->GetGlobalVar(func_name.value());
      // Only rewrite to a PrimFunc if the global definition is still a Relay function.
      need_rewriting = ir_module_->Lookup(global_var)->IsInstance<FunctionNode>();
    } else {
      global_var = GlobalVar(func_name.value());
      global_var->checked_type_ = func->checked_type();
      need_rewriting = true;
    }

    // For illustration only, check if the current target matches the example_target_hook kind,
    // and if so extract the example attribute value.
    int64_t example_attribute_value = 0;
    Optional<Target> opt_current_target = Target::Current();
    if (opt_current_target.defined() &&
        opt_current_target.value()->kind->name == "example_target_hook") {
      example_attribute_value =
          opt_current_target.value()->GetAttr<Integer>("example_attribute").value()->value;
    }

    if (need_rewriting) {
      // The called function is still in Relay form. Convert to TIR.
      tir::Buffer x_buffer = tir::decl_buffer({8}, DataType::Float(32), "x");
      tir::Buffer y_buffer = tir::decl_buffer({8}, DataType::Float(32), "y");
      tir::Buffer out_buffer = tir::decl_buffer({8}, DataType::Float(32));

      tir::Var x_var("x", DataType::Handle());
      tir::Var y_var("y", DataType::Handle());
      tir::Var out_var("out", DataType::Handle());

      Map<String, ObjectRef> dict_attrs;
      dict_attrs.Set("global_symbol", global_var->name_hint);
      dict_attrs.Set("tir.noalias", Bool(true));

      te::Var index("index", DataType::Int(32));
      tir::Sub indexed_sub = tir::Sub(LoadIndex(x_buffer, index), LoadIndex(y_buffer, index));
      if (example_attribute_value > 0) {
        // For illustration only, fold the example attribute into the result.
        indexed_sub = tir::Sub(indexed_sub, FloatImm(DataType::Float(32),
                                                     static_cast<double>(example_attribute_value)));
      }

      tir::Stmt math_body = tir::BufferStore(out_buffer, indexed_sub, {index});
      tir::Stmt math_loop = tir::For(index, 0, 8, tir::ForKind::kSerial, math_body);

      Map<tir::Var, tir::Buffer> buffer_map = {
          {x_var, x_buffer},
          {y_var, y_buffer},
          {out_var, out_buffer},
      };

      tir::PrimFunc replacement_func = tir::PrimFunc({x_var, y_var, out_var}, math_loop, VoidType(),
                                                     buffer_map, DictAttrs(dict_attrs));

      // Switch to TIRToRuntime hook for testing
      Bool tir_to_runtime = func->GetAttr<Bool>("tir_to_runtime").value_or(Bool(false));
      if (tir_to_runtime) {
        replacement_func = WithAttr(replacement_func, ::tvm::attr::kTarget, custom_target_);
      } else {
        replacement_func = WithAttr(replacement_func, ::tvm::attr::kTarget, host_target_);
      }

      ir_module_->Update(global_var, replacement_func);  // Will Add if global_var is new.
    }

    return global_var;
  }

  using MixedModeMutator::VisitExpr_;

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      Expr var = this->VisitExpr(op->var);
      Expr value = this->VisitExpr(op->value);

      if (AsLowerableFunction(value)) {
        // Inline on-the-fly if the let-bound value is lowerable.
        this->memo_[var] = value;
      }
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);

      if (AsLowerableFunction(value)) {
        // The let binding is no longer needed since inlined on-the-fly above.
        this->memo_[expr] = this->VisitExpr(op->body);
      } else {
        Var var = Downcast<Var>(this->VisitExpr(op->var));
        if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  const FunctionNode* AsLowerableFunction(const Expr& expr) {
    if (const auto* function_node = expr.as<FunctionNode>()) {
      auto func_name = function_node->GetAttr<String>(::tvm::attr::kGlobalSymbol);
      if (!func_name.defined()) {
        return nullptr;
      }
      if (func_name != "replace_add_with_subtract") {
        return nullptr;
      }
      return function_node;
    } else if (auto global_var_node = expr.as<GlobalVar>()) {
      return AsLowerableFunction(ir_module_->Lookup(global_var_node.value()));
    } else {
      return nullptr;
    }
  }

  const GlobalVarNode* AsAlreadyLoweredFunction(const Expr& expr) {
    if (auto opt = expr.as<GlobalVar>()) {
      auto global_var = opt.value();
      if (ir_module_->Lookup(global_var).as<tir::PrimFuncNode>()) {
        return global_var.get();
      }
    }
    return nullptr;
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const auto* call = post.as<CallNode>()) {
      GlobalVar new_op;
      if (const auto* function_node = AsLowerableFunction(call->op)) {
        // Add or replace the function with a PrimFunc.
        new_op = ReplaceAddWithSubtractPrimFunc(GetRef<Function>(function_node));
      } else if (const auto* global_var_node = AsAlreadyLoweredFunction(call->op)) {
        // The function has already been rewritten, so we just need to update the call.
        new_op = GetRef<GlobalVar>(global_var_node);
      }
      if (new_op.defined()) {
        // Since we are replacing the Relay function with a call to a TIR function, we must use
        // the call_lowered op.
        CallLoweredAttrs attrs;
        attrs.metadata.Set("relay_attrs", call->attrs);
        ICHECK(call->type_args.empty()) << "lowered functions cannot be polymorphic";
        return CallLowered(std::move(new_op), call->args, std::move(attrs), call->span);
      }
    }

    return post;
  }

 public:
  IRModule ir_module_;
  Target host_target_;
  Target custom_target_;
};

}  // namespace

transform::Pass RelayToTIR() {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        ConvertAddToSubtract relay_to_tir(std::move(ir_module), Target("c"));
        return relay_to_tir.Mutate();
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "RelayToTIR", {});
}

}  // namespace example_target_hooks
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
