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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace relax {

namespace {

bool HasRelaxCallCapabilities(const CallNode* call) {
  auto op = call->op.as<Op>();
  if (!op) return true;
  static auto infer_type_map = Op::GetAttrMap<FInferType>("FInferType");
  static auto legalize_map = Op::GetAttrMap<FLegalize>("FLegalize");
  return infer_type_map.count(op.value()) || legalize_map.count(op.value());
}

class PrimExprComputeInjector : public ExprMutator {
 public:
  IRModule Finalize() const { return builder_->Finalize(); }

  Type VisitExprDepTypeField(const Type& ty) final { return ty; }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* op) final {
    Call call = ffi::GetRef<Call>(op);
    if (auto prim_expr = call.as<PrimExpr>()) {
      if (call->op.as<Op>() && !HasRelaxCallCapabilities(op)) {
        return LiftPrimValue(prim_expr.value());
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

#define RELAX_LIFT_PRIM_EXPR(OP)                                            \
  Expr VisitExpr_(const OP* op) final {                                     \
    return LiftPrimValue(ffi::GetRef<Expr>(op).as_or_throw<PrimExpr>());    \
  }

  RELAX_LIFT_PRIM_EXPR(tirx::BufferLoadNode);
  RELAX_LIFT_PRIM_EXPR(tirx::AddNode);
  RELAX_LIFT_PRIM_EXPR(tirx::SubNode);
  RELAX_LIFT_PRIM_EXPR(tirx::MulNode);
  RELAX_LIFT_PRIM_EXPR(tirx::DivNode);
  RELAX_LIFT_PRIM_EXPR(tirx::ModNode);
  RELAX_LIFT_PRIM_EXPR(tirx::FloorDivNode);
  RELAX_LIFT_PRIM_EXPR(tirx::FloorModNode);
  RELAX_LIFT_PRIM_EXPR(tirx::MinNode);
  RELAX_LIFT_PRIM_EXPR(tirx::MaxNode);
  RELAX_LIFT_PRIM_EXPR(tirx::EQNode);
  RELAX_LIFT_PRIM_EXPR(tirx::NENode);
  RELAX_LIFT_PRIM_EXPR(tirx::LTNode);
  RELAX_LIFT_PRIM_EXPR(tirx::LENode);
  RELAX_LIFT_PRIM_EXPR(tirx::GTNode);
  RELAX_LIFT_PRIM_EXPR(tirx::GENode);
  RELAX_LIFT_PRIM_EXPR(tirx::AndNode);
  RELAX_LIFT_PRIM_EXPR(tirx::OrNode);
  RELAX_LIFT_PRIM_EXPR(tirx::CastNode);
  RELAX_LIFT_PRIM_EXPR(tirx::NotNode);
  RELAX_LIFT_PRIM_EXPR(tirx::SelectNode);
  RELAX_LIFT_PRIM_EXPR(tirx::RampNode);
  RELAX_LIFT_PRIM_EXPR(tirx::BroadcastNode);
  RELAX_LIFT_PRIM_EXPR(tirx::ShuffleNode);
  RELAX_LIFT_PRIM_EXPR(tvm::IntImmNode);
  RELAX_LIFT_PRIM_EXPR(tvm::FloatImmNode);
  RELAX_LIFT_PRIM_EXPR(tirx::StringImmNode);

#undef RELAX_LIFT_PRIM_EXPR

  Expr VisitExpr_(const ShapeExprNode* op) final { return ffi::GetRef<Expr>(op); }

  Expr LiftPrimValue(const PrimExpr& node) {
    if (node->IsInstance<tirx::IntImmNode>() || node->IsInstance<VarNode>()) {
      return node;
    }

    tvm::PrimType ret_ty = node.ty();
    auto param_vars = tirx::UndefinedVars(node);
    tirx::Stmt body =
        tirx::Evaluate(tvm::Call(node.ty(), tirx::builtin::ret(), {node}).as_or_throw<PrimExpr>());

    tirx::PrimFunc func(param_vars, body, ret_ty, {},
                        DictAttrs({{tirx::attr::kIsHostFunc, true}, {tvm::attr::kSTir, true}}));
    func = s_tir::RenewDefs(func);

    auto callee = builder_->AddFunction(func, "compute_symbolic_expr");

    return Call(ret_ty, callee, param_vars.Map([](const tirx::Var& tir_var) -> relax::Expr {
      return tir_var.as_or_throw<PrimExpr>();
    }));
  }
};

}  // namespace

namespace transform {

Pass ComputePrimValue() {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    PrimExprComputeInjector mutator;

    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto updated = mutator(func.value()).as_or_throw<Function>();
        if (!updates.same_as(base_func)) {
          updates->Add(gvar, updated);
        }
      }
    }

    if (updates->functions.size()) {
      auto write_ptr = mod.CopyOnWrite();
      write_ptr->Update(updates);
      write_ptr->Update(mutator.Finalize());
    }

    return mod;
  };
  return CreateModulePass(pass_func, 0, "ComputePrimValue", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ComputePrimValue", ComputePrimValue);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
