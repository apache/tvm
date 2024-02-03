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
 * \file src/contrib/msc/core/transform/inline_params.cc
 * \brief Pass for inline Exprs.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace relax {

using namespace tvm::contrib::msc;

/*!
 * \brief Inline the exprs
 */
class ParamsInliner : public ExprMutator {
 public:
  explicit ParamsInliner(IRModule ctx_module, const String& entry_name) : ExprMutator(ctx_module) {
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
        Array<String> attrs;
        for (const auto& p : Downcast<Function>(func)->params) {
          auto struct_info = GetStructInfo(p);
          if (struct_info->IsInstance<ShapeStructInfoNode>()) {
            continue;
          }
          if (struct_info->IsInstance<FuncStructInfoNode>()) {
            const auto& optype_opt = func->GetAttr<runtime::String>(msc_attr::kOptype);
            ICHECK(optype_opt.defined())
                << "Can not find attr " << msc_attr::kOptype << " form extern func";
            extern_types_.Set(p, optype_opt.value());
            continue;
          }
          if (const auto* tuple_info = struct_info.as<TupleStructInfoNode>()) {
            Array<StructInfo> new_fields;
            for (const auto& i : tuple_info->fields) {
              if (i->IsInstance<TensorStructInfoNode>()) {
                new_fields.push_back(i);
              } else if (const auto& p_info = i.as<PrimStructInfoNode>()) {
                ICHECK(p_info->value.defined()) << "PrimStructInfo with undefined prim value " << i;
                attrs.push_back(StringUtils::ToString(p_info->value.value()));
              }
            }
            if (new_fields.size() < tuple_info->fields.size()) {
              p->struct_info_ = TupleStructInfo(new_fields, tuple_info->span);
            }
          }
          new_params.push_back(p);
        }
        if (new_params.size() == Downcast<Function>(func)->params.size()) {
          continue;
        }
        const auto& new_func = Downcast<Function>(VisitExpr(func));
        Map<String, ObjectRef> func_attrs = new_func->attrs->dict;
        if (attrs.size() > 0) {
          func_attrs.Set(msc_attr::kOpattrs, attrs);
        }
        auto updated_func = Function(new_params, new_func->body, new_func->ret_struct_info,
                                     new_func->is_pure, DictAttrs(func_attrs), new_func->span);
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
    bool has_inline = false;
    for (const auto& a : call_node->args) {
      auto struct_info = GetStructInfo(a);
      if (a->IsInstance<VarNode>() && struct_info->IsInstance<FuncStructInfoNode>()) {
        ICHECK(extern_types_.count(a)) << "Can not find extern type of " << a;
        new_args.push_back(ExternFunc(extern_types_[a]));
        has_inline = true;
      } else if (call_node->op->IsInstance<GlobalVarNode>() && a->IsInstance<ExternFuncNode>()) {
        has_inline = true;
      } else if (a->IsInstance<VarNode>() && struct_info->IsInstance<ShapeStructInfoNode>()) {
        const auto& shape_opt = Downcast<ShapeStructInfo>(GetStructInfo(a))->values;
        ICHECK(shape_opt.defined()) << "Expected shape defined, get " << a;
        new_args.push_back(ShapeExpr(shape_opt.value()));
        has_inline = true;
      } else if (call_node->op->IsInstance<GlobalVarNode>() && a->IsInstance<ShapeExprNode>()) {
        has_inline = true;
      } else if (call_node->op->IsInstance<GlobalVarNode>() && a->IsInstance<TupleNode>()) {
        const auto& tuple = Downcast<Tuple>(a);
        Array<Expr> new_fields;
        Array<StructInfo> new_infos;

        for (const auto& f : tuple->fields) {
          if (f->IsInstance<VarNode>()) {
            new_fields.push_back(f);
            new_infos.push_back(GetStructInfo(f));
          }
        }
        if (new_fields.size() == tuple->fields.size()) {
          new_args.push_back(a);
        } else {
          const auto& new_tuple = Tuple(new_fields, tuple->span);
          new_tuple->struct_info_ = TupleStructInfo(new_infos);
          new_args.push_back(new_tuple);
        }
      } else {
        new_args.push_back(a);
      }
    }
    if (!has_inline) {
      ExprMutator::VisitBinding_(binding, call_node);
    } else if (call_node->op->IsInstance<OpNode>()) {
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
  Map<Expr, String> extern_types_;
};

IRModule InlineParams(IRModule mod, const String& entry_name) {
  return ParamsInliner(mod, entry_name).Bind();
}

namespace transform {

Pass InlineParams(const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::InlineParams(m, entry_name); };
  return CreateModulePass(pass_func, 0, "InlineParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.InlineParams").set_body_typed(InlineParams);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
