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
 * \file src/contrib/msc/core/transform/fuse_shape.cc
 * \brief Pass for fuse ShapeExpr.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Bind ShapeExpr to Reshape
 */
class ShapeBinder : public ExprMutator {
 public:
  explicit ShapeBinder(IRModule ctx_module, const String& entry_name) : ExprMutator(ctx_module) {
    mod_ = ctx_module;
    entry_name_ = entry_name;
  }

  IRModule Bind() {
    // update global functions
    GlobalVar main_var;
    for (const auto& [gv, func] : mod_->functions) {
      if (gv->name_hint == entry_name_) {
        main_var = gv;
        continue;
      }
      if (func->IsInstance<FunctionNode>()) {
        Array<Var> new_params;
        for (const auto& p : Downcast<Function>(func)->params) {
          auto struct_info = GetStructInfo(p);
          if (struct_info->IsInstance<ShapeStructInfoNode>()) {
            continue;
          }
          new_params.push_back(p);
        }
        if (new_params.size() == Downcast<Function>(func)->params.size()) {
          continue;
        }
        const auto& new_func = Downcast<Function>(VisitExpr(func));
        auto updated_func = Function(new_params, new_func->body, new_func->ret_struct_info,
                                     new_func->is_pure, new_func->attrs, new_func->span);
        builder_->UpdateFunction(gv, updated_func);
      }
    }
    // update main
    ICHECK(main_var.defined()) << "Can not find entry func " << entry_name_;
    const auto& new_func = Downcast<Function>(VisitExpr(mod_->Lookup(entry_name_)));
    builder_->UpdateFunction(main_var, new_func);
    return builder_->GetContextIRModule();
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    Array<Expr> new_args;
    for (const auto& a : call_node->args) {
      auto struct_info = GetStructInfo(a);
      if (a->IsInstance<VarNode>() && struct_info->IsInstance<ShapeStructInfoNode>()) {
        continue;
      }
      if (call_node->op->IsInstance<GlobalVarNode>() && a->IsInstance<ShapeExprNode>()) {
        continue;
      }
      new_args.push_back(a);
    }
    if (new_args.size() == call_node->args.size()) {
      ExprMutator::VisitBinding_(binding, call_node);
    } else if (const auto* op_node = call_node->op.as<OpNode>()) {
      ICHECK(op_node->name == "relax.reshape" || op_node->name == "relax.image.resize2d")
          << "Expect ShapeExpr consumer as reshape or image.resize2d, get "
          << GetRef<Call>(call_node);
      const auto& opt_shape = Downcast<ShapeStructInfo>(GetStructInfo(call_node->args[1]))->values;
      ICHECK(opt_shape.defined()) << "Expected shape defined, get " << call_node->args[1];
      new_args.push_back(ShapeExpr(opt_shape.value()));
      const auto& new_call =
          Call(call_node->op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      ReEmitBinding(binding, builder_->Normalize(new_call));
    } else if (const auto* gv_node = call_node->op.as<GlobalVarNode>()) {
      const auto& func_info = Downcast<FuncStructInfo>(gv_node->struct_info_);
      Array<StructInfo> params_info;
      for (const auto& a : new_args) {
        ICHECK(a->struct_info_.defined())
            << "Global func argument without defined struct info " << a;
        params_info.push_back(Downcast<StructInfo>(a->struct_info_.value()));
      }
      call_node->op->struct_info_ =
          FuncStructInfo(params_info, func_info->ret, func_info->purity, func_info->span);
      const auto& new_call =
          Call(call_node->op, new_args, call_node->attrs, call_node->sinfo_args, call_node->span);
      ReEmitBinding(binding, builder_->Normalize(new_call));
    } else {
      LOG_FATAL << "Unexpected shape consumer " << GetRef<Call>(call_node);
    }
  }

 private:
  IRModule mod_;
  String entry_name_;
};

IRModule BindShape(IRModule mod, const String& entry_name) {
  return ShapeBinder(mod, entry_name).Bind();
}

namespace transform {

Pass BindShape(const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::BindShape(m, entry_name); };
  return CreateModulePass(pass_func, 0, "BindShape", {});
}

TVM_REGISTER_GLOBAL("relax.transform.BindShape").set_body_typed(BindShape);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
