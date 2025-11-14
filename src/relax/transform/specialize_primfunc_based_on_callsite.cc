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

#include <tvm/node/serialization.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/index_map.h>

#include <tuple>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

using tvm::tir::Buffer;

static ffi::Array<PrimExpr> GetShapeFromTensorStructInfo(const TensorStructInfo& tensor_sinfo) {
  auto shape = tensor_sinfo->GetShape();
  ICHECK(shape.defined());
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
        relax::Function update_func = Downcast<Function>(VisitExpr(func));
        updates_->Add(gv, update_func);
      }
    }
    mod_.CopyOnWrite()->Update(updates_);
    return mod_;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) override {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op == call_tir_op) {
      return SpecializeTirPrimFunc(call);
    }
    return call;
  }

 private:
  Expr SpecializeTirPrimFunc(Call call) {
    auto gv = Downcast<GlobalVar>(call->args[0]);
    auto pfunc = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
    auto args = Downcast<Tuple>(call->args[1])->fields;
    ffi::Map<tir::Var, ffi::Variant<Buffer, PrimExpr>> param_map;

    for (size_t i = 0; i < args.size(); ++i) {
      auto sinfo = GetStructInfo(args[i]);
      CHECK(sinfo->IsInstance<TensorStructInfoNode>())
          << "Expected Tensor struct Info for call :" << call->op;
      auto tensor_sinfo = Downcast<TensorStructInfo>(sinfo);
      CHECK(tensor_sinfo->shape.defined()) << "Shape undefined for call:" << call->args[0];
      ffi::String scope = "global";
      if (tensor_sinfo->vdevice.defined()) {
        scope = tensor_sinfo->vdevice.value()->memory_scope;
      }
      ffi::String name;
      if (args[i]->IsInstance<relax::VarNode>()) {
        name = Downcast<Var>(args[i])->name_hint();
      } else {
        name = std::string({static_cast<char>('A' + i)});
      }

      const Buffer& buffer = tir::decl_buffer(GetShapeFromTensorStructInfo(tensor_sinfo),
                                              tensor_sinfo->dtype, name, scope);
      param_map.Set(pfunc->params[i], buffer);
    }
    ffi::String scope = "global";
    auto out_sinfo = call->sinfo_args[0];
    if (out_sinfo->IsInstance<TensorStructInfoNode>()) {
      auto sinfo = Downcast<TensorStructInfo>(out_sinfo);
      if (sinfo->vdevice.defined()) {
        scope = sinfo->vdevice.value()->memory_scope;
      }
      const Buffer& buffer =
          tir::decl_buffer(GetShapeFromTensorStructInfo(sinfo), sinfo->dtype, "ret_val", scope);
      param_map.Set(pfunc->params[pfunc->params.size() - 1], buffer);
    } else {
      ICHECK(out_sinfo->IsInstance<TupleStructInfoNode>())
          << "Expect output struct info of call_tir to be either TupleStructInfo or "
             "TensorStructInfo, but got "
          << out_sinfo;

      const auto& tuple_sinfo = Downcast<TupleStructInfo>(out_sinfo);
      ffi::Array<StructInfo> sinfo_fields;
      int index = 0;
      for (const auto& si : tuple_sinfo->fields) {
        ICHECK(si->IsInstance<TensorStructInfoNode>())
            << "Fields of TupleStructInfo must be TensorStructInfo for call_tir "
               "output structinfo, but got "
            << si;
        auto sinfo = Downcast<TensorStructInfo>(si);
        if (sinfo->vdevice.defined()) {
          scope = sinfo->vdevice.value()->memory_scope;
        }
        const Buffer& buffer =
            tir::decl_buffer(GetShapeFromTensorStructInfo(sinfo), sinfo->dtype, "ret_val", scope);
        param_map.Set(pfunc->params[args.size() + index], buffer);
        index++;
      }
    }

    auto new_pfunc = Specialize(pfunc, param_map);
    for (const auto& [var, buffer] : new_pfunc->buffer_map) {
      auto* ptr = buffer->data->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    }
    auto new_prim_func = WithAttr(new_pfunc, "scoped", Integer(1));
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
