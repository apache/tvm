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
 * \file lambda_expr.cc
 * \brief Implementation of LambdaExpr, a reified lambda shared by IR consumers.
 */

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/type.h>

#include <utility>
#include <vector>

namespace tvm {

namespace {

ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> LambdaExprVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LambdaExprNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->vars); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  return visitor->VisitExpected(self->body);
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>> LambdaExprMutate(ffi::StructuralMutatorObj* mutator,
                                                           ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LambdaExprNode>(value);
  // Parameters shadow substitutions from the surrounding expression. Save and restore
  // their remaps so a binder rewrite stays local to this lambda's body.
  std::vector<ffi::Any> saved;
  for (const Var& var : self->vars) {
    auto previous = mutator->VarRemapGetExpected(var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(previous);
    saved.push_back(std::move(previous).value());
  }
  auto result = [&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    try {
      for (const Var& var : self->vars) {
        auto cleared = mutator->VarRemapSetExpected(var, nullptr);
        TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(cleared);
      }
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
          ffi::UnchangedOr<ffi::Array<Var>>, mapped_vars,
          mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                     [&]() { return mutator->MutateExpected(self->vars); }));
      auto vars = std::move(mapped_vars).ValueOrUnchanged(self->vars);
      for (size_t i = 0; i < vars.size(); ++i) {
        auto remap = mutator->VarRemapSetExpected(self->vars[i], vars[i]);
        TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
      }
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_body,
                                        mutator->MutateExpected(self->body));
      Expr body = std::move(mapped_body).ValueOrUnchanged(self->body);
      ffi::Array<Type> param_types;
      for (const Var& var : vars) param_types.push_back(var->ty);
      Type ty = FuncType(param_types, body->ty);
      if (vars.same_as(self->vars) && body.same_as(self->body) &&
          ffi::StructuralEqual()(ty, self->ty)) {
        return ffi::Unchanged();
      }
      auto copy = ffi::make_object<LambdaExprNode>(*self);
      copy->vars = std::move(vars);
      copy->body = std::move(body);
      copy->ty = std::move(ty);
      return ffi::Any(std::move(copy));
    } catch (const ffi::Error& error) {
      return ffi::Unexpected(error);
    } catch (const std::exception& error) {
      return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
    }
  }();
  for (size_t i = 0; i < self->vars.size(); ++i) {
    auto restored = mutator->VarRemapSetExpected(self->vars[i], saved[i]);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(restored);
  }
  return result;
}

// Substitute simultaneously. Every expression delimits the lifetime of definitions
// introduced by its structural children. Non-expression binding blocks share their
// enclosing expression's scope, while sibling expressions cannot leak remaps.
class LambdaApplier : public ExprMutator {
 public:
  using ExprMutator::Mutate;
  explicit LambdaApplier(ffi::Map<Var, Expr> arguments) : arguments_(std::move(arguments)) {}

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode mode) final {
    if (auto var = value.as<Var>()) {
      if (def_region_kind() != kTVMFFIDefRegionKindNone) {
        return ffi::Any(Freshen(var.value()));
      }
      if (VarRemapGet(var.value()) == nullptr) {
        if (auto replacement = arguments_.Get(var.value())) return ffi::Any(replacement.value());
      }
      return ExprMutator::Mutate(value, mode);
    }
    if (!value.as<Expr>()) return ExprMutator::Mutate(value, mode);

    const size_t scope_start = saved_remaps_.size();
    auto restore = [&]() {
      while (saved_remaps_.size() > scope_start) {
        const auto& [var, previous] = saved_remaps_.back();
        VarRemapSet(var, previous);
        saved_remaps_.pop_back();
      }
    };
    try {
      auto result = MutateExpr(value, mode);
      restore();
      return result;
    } catch (...) {
      restore();
      throw;
    }
  }

 private:
  Var Freshen(const Var& var) {
    // Preserve runtime subtypes such as dataflow variables without depending on
    // their dialect. Rewrite dependent types before installing the new binder.
    Type ty = WithDefRegionKind(kTVMFFIDefRegionKindNone, [&]() {
      return Mutate(var->ty, InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<Type>>()
          .ValueOrUnchanged(var->ty);
    });
    static ffi::reflection::TypeAttrColumn shallow_copy(ffi::reflection::type_attr::kShallowCopy);
    auto copy = shallow_copy[var->type_index()].cast<ffi::Function>()(var).cast<Var>();
    TVM_FFI_CHECK(copy.type_index() == var.type_index() && !copy.same_as(var), TypeError)
        << "Variable shallow copy must preserve its runtime type and return a fresh object";
    const_cast<VarNode*>(copy.get())->ty = std::move(ty);
    saved_remaps_.emplace_back(var, VarRemapGet(var));
    VarRemapSet(var, copy);
    return copy;
  }

  UnchangedOr<ffi::Any> MutateExpr(ffi::AnyView value, InplaceMode mode) {
    if (auto lambda = value.as<LambdaExpr>()) {
      ffi::Array<Var> vars;
      for (const Var& var : lambda.value()->vars) vars.push_back(Freshen(var));
      Expr body = Mutate(lambda.value()->body).ValueOrUnchanged(lambda.value()->body);
      LambdaExpr result(vars, body);
      result->span = lambda.value()->span;
      return ffi::Any(result);
    }
    if (auto let = value.as<prim::Let>()) {
      PrimExpr initial = Mutate(let.value()->value).ValueOrUnchanged(let.value()->value);
      Var var = Freshen(let.value()->var);
      PrimExpr body = Mutate(let.value()->body).ValueOrUnchanged(let.value()->body);
      return ffi::Any(prim::Let(var, initial, body, let.value()->span));
    }
    return ExprMutator::Mutate(value, mode);
  }

  ffi::Map<Var, Expr> arguments_;
  std::vector<std::pair<Var, ffi::Any>> saved_remaps_;
};

bool ContainsMissingType(const Type& type) {
  bool missing = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(type, [&](const MissingTypeNode*) {
    missing = true;
    return ffi::WalkResult::Skip();
  });
  return missing;
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  StagingExprNode::RegisterReflection();
  LambdaExprNode::RegisterReflection();
  refl::TypeAttrDef<LambdaExprNode>()
      .def("__s_equal__",
           [](LambdaExpr self, LambdaExpr other,
              ffi::TypedFunction<bool(ffi::AnyView, ffi::AnyView, int, ffi::AnyView)> equal) {
             // The derived signature may refer to these parameters. Simple binds
             // only the parameters themselves, preserving free variables in types.
             return equal(self->vars, other->vars, kTVMFFIDefRegionKindSimple, "vars") &&
                    equal(self->body, other->body, kTVMFFIDefRegionKindNone, "body") &&
                    equal(self->ty, other->ty, kTVMFFIDefRegionKindNone, "ty");
           })
      .def("__s_hash__",
           [](LambdaExpr self, int64_t init,
              ffi::TypedFunction<int64_t(ffi::AnyView, int64_t, int)> hash) {
             int64_t result = hash(self->vars, init, kTVMFFIDefRegionKindSimple);
             result = hash(self->body, result, kTVMFFIDefRegionKindNone);
             return hash(self->ty, result, kTVMFFIDefRegionKindNone);
           })
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&LambdaExprVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&LambdaExprMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&LambdaExprMutate>());
}

Expr LambdaExprNode::Apply(const ffi::Array<Expr>& indices) const {
  TVM_FFI_CHECK_EQ(indices.size(), vars.size(), ValueError) << "LambdaExpr Apply arity mismatch";
  ffi::Map<Var, Expr> vmap;
  for (size_t i = 0; i < vars.size(); ++i) {
    // A later parameter type can depend on the arguments of earlier parameters.
    Type expected = LambdaApplier(vmap)
                        .Mutate(vars[i]->ty, InplaceMode::kDisallow)
                        .as_or_throw<UnchangedOr<Type>>()
                        .ValueOrUnchanged(vars[i]->ty);
    TVM_FFI_CHECK(
        indices[i]->ty.as<MissingTypeNode>() || ffi::StructuralEqual()(indices[i]->ty, expected),
        TypeError)
        << "LambdaExpr Apply argument type mismatch at index " << i;
    vmap.Set(vars[i], indices[i]);
  }
  return LambdaApplier(std::move(vmap)).Mutate(body).ValueOrUnchanged(body);
}

LambdaExpr::LambdaExpr(ffi::Array<Var> vars, Expr body) : StagingExpr(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<LambdaExprNode>(std::move(body));
  ffi::Array<Type> types;
  for (size_t i = 0; i < vars.size(); ++i) {
    TVM_FFI_CHECK(!ContainsMissingType(vars[i]->ty), TypeError)
        << "LambdaExpr parameters require explicit non-Missing types";
    for (size_t j = 0; j < i; ++j) {
      TVM_FFI_CHECK(!vars[i].same_as(vars[j]), ValueError)
          << "LambdaExpr parameters must be distinct variables";
    }
    types.push_back(vars[i]->ty);
  }
  n->ty = FuncType(types, n->body->ty);
  n->vars = std::move(vars);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.LambdaExpr",
                        [](ffi::Array<Var> vars, Expr body) { return LambdaExpr(vars, body); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.LambdaExprApply", [](LambdaExpr body, ffi::Array<Expr> indices) {
    return body->Apply(indices);
  });
}

}  // namespace tvm
