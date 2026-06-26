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
 * \file src/relax/transform/specialize_tir_params.cc
 * \brief Update PrimFunc buffers based on updated scope (or structure) info.
 */

#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/index_map.h>

#include <tuple>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

using tvm::tirx::Buffer;

static ffi::Array<PrimExpr> GetShapeFromTensorType(const TensorType& tensor_ty) {
  auto shape = tensor_ty->GetShape();
  TVM_FFI_ICHECK(shape.defined());
  return shape.value();
}

class SpecializeTIRCallArgs : ExprMutator {
 public:
  IRModule Run(IRModule mod) {
    mod_ = mod;
    for (const auto& [gv, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionNode>()) {
        const auto& base_func = mod->Lookup(gv);
        // Only non primitive relax functions
        if (base_func->HasNonzeroAttr(attr::kPrimitive)) {
          continue;
        }
        relax::Function update_func = VisitExpr(func).as_or_throw<Function>();
        updates_->Add(gv, update_func);
      }
    }
    mod_.CopyOnWrite()->Update(updates_);
    return mod_;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) override {
    auto call = ExprMutator::VisitExpr_(call_node).as_or_throw<Call>();
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op == call_tir_op) {
      return SpecializeTirPrimFunc(call);
    }
    return call;
  }

 private:
  Expr SpecializeTirPrimFunc(Call call) {
    auto gv = call->args[0].as_or_throw<GlobalVar>();
    auto pfunc = mod_->Lookup(gv).as_or_throw<tirx::PrimFunc>();
    auto args = call->args[1].as_or_throw<Tuple>()->fields;
    ffi::Map<tirx::Var, ffi::Variant<Buffer, PrimExpr>> param_map;

    for (size_t i = 0; i < args.size(); ++i) {
      auto ty = GetType(args[i]);
      TVM_FFI_ICHECK(ty->IsInstance<TensorTypeNode>())
          << "Expected Tensor struct Info for call :" << call->op;
      auto tensor_ty = ty.as_or_throw<TensorType>();
      TVM_FFI_ICHECK(tensor_ty->shape.defined()) << "Shape undefined for call:" << call->args[0];
      ffi::String scope = "global";
      if (tensor_ty->vdevice.defined()) {
        scope = tensor_ty->vdevice.value()->memory_scope;
      }
      ffi::String name;
      if (args[i]->IsInstance<relax::VarNode>()) {
        name = args[i].as_or_throw<Var>()->name_hint();
      } else {
        name = std::string({static_cast<char>('A' + i)});
      }

      const Buffer& buffer = tirx::decl_buffer(GetShapeFromTensorType(tensor_ty),
                                               tensor_ty->dtype.value(), name, scope);
      param_map.Set(pfunc->params[i], buffer);
    }
    ffi::String scope = "global";
    auto out_ty = call->ty_args[0];
    if (out_ty->IsInstance<TensorTypeNode>()) {
      auto ty = out_ty.as_or_throw<TensorType>();
      if (ty->vdevice.defined()) {
        scope = ty->vdevice.value()->memory_scope;
      }
      const Buffer& buffer =
          tirx::decl_buffer(GetShapeFromTensorType(ty), ty->dtype.value(), "ret_val", scope);
      param_map.Set(pfunc->params[pfunc->params.size() - 1], buffer);
    } else {
      TVM_FFI_ICHECK(out_ty->IsInstance<TupleTypeNode>())
          << "Expect output type of call_tir to be either TupleType or "
             "TensorType, but got "
          << out_ty;

      const auto& tuple_ty = out_ty.as_or_throw<TupleType>();
      ffi::Array<Type> ty_fields;
      int index = 0;
      for (const auto& si : tuple_ty->fields) {
        TVM_FFI_ICHECK(si->IsInstance<TensorTypeNode>())
            << "Fields of TupleType must be TensorType for call_tir "
               "output structinfo, but got "
            << si;
        auto ty = si.as_or_throw<TensorType>();
        if (ty->vdevice.defined()) {
          scope = ty->vdevice.value()->memory_scope;
        }

        const Buffer& buffer = tirx::decl_buffer(GetShapeFromTensorType(ty), ty->dtype.value(),
                                                 "ret_val_" + std::to_string(index), scope);
        param_map.Set(pfunc->params[args.size() + index], buffer);
        index++;
      }
    }

    auto new_pfunc = Specialize(pfunc, param_map);
    for (const auto& [var, buffer] : new_pfunc->buffer_map) {
      auto* ptr = buffer->data->type_annotation.as<PointerTypeNode>();
      TVM_FFI_ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    }
    auto new_prim_func = WithAttr(new_pfunc, "scoped", static_cast<int64_t>(1));
    updates_->Add(gv, new_prim_func);
    return call;
  }
  IRModule mod_;
  IRModule updates_;
};

namespace transform {

Pass SpecializePrimFuncBasedOnCallSite() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return relax::SpecializeTIRCallArgs().Run(mod);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"SpecializePrimFuncBasedOnCallSite",
                          /*required=*/{});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.SpecializePrimFuncBasedOnCallSite",
                        SpecializePrimFuncBasedOnCallSite);
}
}  // namespace transform
}  // namespace relax
}  // namespace tvm
