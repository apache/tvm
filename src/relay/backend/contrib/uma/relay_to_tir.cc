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
 * \file relay/backend/contrib/uma/codegen.cc
 *
 * \brief this file contains the target hooks for the Universal Modular Accelerator Interface (UMA).
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {
namespace uma {

// TODO(@mjklaiber, @manupa-arm, @areusch) move this to include
/*!
 * \brief This mutator outlines functions that are marked with a named
 * "Compiler" attribute. Functions that do not match this condition remain
 * unaltered.
 */
class OutlineCompilerFunctionsMutator : public MixedModeMutator {
 public:
  explicit OutlineCompilerFunctionsMutator(const IRModule& mod, const std::string& compiler_name)
      : mod_(mod), compiler_name_(compiler_name) {}

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      Expr var = this->VisitExpr(op->var);
      Expr value = this->VisitExpr(op->value);

      // Outlineable function no longer needs let binding
      if (this->CanOutlineExpr(value)) {
        this->memo_[var] = value;
      }
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);

      // Drop the let binding
      if (this->CanOutlineExpr(value)) {
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

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    Call call = Downcast<Call>(post);
    if (CanOutlineExpr(call->op)) {
      Function func = Downcast<Function>(call->op);
      auto gv_name = func->GetAttr<String>("global_symbol").value_or("");
      ICHECK_NE(gv_name, "")
          << "Function to be outlined must have global_symbol attribute, but didn't.";
      GlobalVar gv(gv_name);
      if (func->checked_type_.defined()) {
        gv->checked_type_ = func->checked_type();
      }
      mod_->Update(gv, func);
      return Call(gv, call->args, call->attrs, call->type_args);
    }
    return post;
  }

 private:
  /*!
   * \brief Check if the expr is a function and has the same
   * compiler name as compiler_name_.
   *
   * \param expr The input expr.
   * \return True if is outlineable else False.
   */
  bool CanOutlineExpr(const Expr& expr) {
    if (!expr->IsInstance<FunctionNode>()) {
      return false;
    }
    Function func = Downcast<Function>(expr);
    auto compiler = func->GetAttr<String>(attr::kCompiler);
    if (!compiler.defined()) {
      return false;
    }
    if (compiler != compiler_name_) {
      return false;
    }
    return true;
  }

  /*! \brief The module that the pass will run on. */
  IRModule mod_;
  /*! \brief The name of the compiler to enable outlining on external functions for. */
  std::string compiler_name_;
};

/*!
 * \brief A pass to outline compiler specific functions.
 */
tvm::transform::Pass OutlineCompilerFunctions(const std::string& compiler_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule mod, transform::PassContext ctx) {
        GlobalVar gv = mod->GetGlobalVar("main");
        Function main_func = Downcast<Function>(mod->Lookup("main"));
        auto new_main_body =
            OutlineCompilerFunctionsMutator(mod, compiler_name).VisitExpr(main_func->body);
        if (!new_main_body.same_as(main_func->body)) {
          Function new_main_func = WithFields(main_func, main_func->params, new_main_body);
          mod->Update(gv, new_main_func);
        }
        return mod;
      };
  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "relay.backend.contrib.uma.OutlineCompilerFunctions", {});
}

TVM_REGISTER_GLOBAL("relay.ext.uma.OutlineCompilerFunctions")
    .set_body_typed(OutlineCompilerFunctions);

/*!
 * \brief This pass will lower UMA functions in a Relay module to scheduled TIR prim functions.
 */
tvm::transform::Pass RelayToTIR(String target_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule ir_module, transform::PassContext pass_context) {
        auto relay_to_tir_pf =
            tvm::runtime::Registry::Get("relay.ext.uma." + target_name + ".relay_to_tir");
        ICHECK(relay_to_tir_pf);
        ir_module = (*relay_to_tir_pf)(ir_module);
        return ir_module;
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "relay.contrib.uma.RelayToTIR", {});
}

}  // namespace uma
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
