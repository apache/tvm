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
 * \file ir_utils.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include "ir_utils.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {
class IRConvertSSA final : public StmtExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(IRConvertSSA, StmtExprMutator)
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  Function VisitFunction(Function func);
  IRModule VisitIRModule(IRModule mod);

 private:
  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) final;
  Var DefineVar(Var var);
  ffi::Optional<Var> GetVarRemap(const Var& var);
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const WhileNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) final;
  static Var MakeNewVar(const Var& old_var);
  void SetVarRemap(const Var& old_var, const Var& new_var);

  // Lexical visibility ends with each scope, but definitions stay unique across the module.
  ScopeStack<std::unordered_map<const VarNode*, Var>> scope_;
  std::unordered_set<const VarNode*> defined_;
};

Function IRConvertSSA::VisitFunction(Function func) {
  return scope_.WithNewScope([&]() -> Function {
    std::unordered_set<const VarNode*> parameter_symbols;
    // Define explicit parameters before the symbolic values in their types.
    for (const Var& param : func->params) {
      DefineVar(param);
      parameter_symbols.insert(param.get());
    }
    for (const Var& param : func->params) {
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          param->ty, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
            if (!parameter_symbols.count(var.get())) {
              parameter_symbols.insert(var.get());
              DefineVar(var);
            }
            return ffi::WalkResult::Advance();
          });
    }
    auto params = func->params.Map([&](const Var& var) {
      Var mapped = GetVarRemap(var).value_or(var);
      Type type = Mutate(var->ty, InplaceMode::kDisallow)
                      .as_or_throw<UnchangedOr<Type>>()
                      .ValueOrUnchanged(var->ty);
      if (!type.same_as(mapped->ty)) {
        mapped = mapped.CopyWithType(type);
        SetVarRemap(var, mapped);
      }
      return mapped;
    });

    auto body_result = Mutate(func->body, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);

    if (!params.same_as(func->params) || !body_unchanged) {
      auto* node = func.CopyOnWrite();
      node->params = std::move(params);
      node->body = std::move(body);
      node->ty = node->func_type_annotation();
    }
    return func;
  });
}

UnchangedOr<Expr> IRConvertSSA::Mutate_(const VarNode* op, InplaceMode inplace_mode) {
  Var var = ffi::GetRef<Var>(op);
  if (def_region_kind() != kTVMFFIDefRegionKindNone) {
    if (def_region_kind() == kTVMFFIDefRegionKindPattern) {
      if (auto mapped = GetVarRemap(var)) return mapped.value();
    }
    return DefineVar(var);
  }
  Var mapped = GetVarRemap(var).value_or(var);
  if (!mapped.same_as(var)) return mapped;
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> IRConvertSSA::Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) {
  ffi::Any previous_remap = VarRemapGet(op->var);
  PrimExpr result = scope_.WithNewScope([&] {
    return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
  });
  VarRemapSet(op->var, previous_remap);
  return result;
}

Var IRConvertSSA::DefineVar(Var var) {
  Type type = WithDefRegionKind(kTVMFFIDefRegionKindNone, [&] {
    return Mutate(var->ty, InplaceMode::kDisallow)
        .as_or_throw<UnchangedOr<Type>>()
        .ValueOrUnchanged(var->ty);
  });
  Var result = defined_.count(var.get()) ? MakeNewVar(var) : var;
  defined_.insert(var.get());
  if (!type.same_as(result->ty)) result = result.CopyWithType(type);
  SetVarRemap(var, result);
  return result;
}

ffi::Optional<Var> IRConvertSSA::GetVarRemap(const Var& var) {
  for (auto it = scope_.rbegin(); it != scope_.rend(); ++it) {
    if (auto found = it->find(var.get()); found != it->end()) return found->second;
  }
  return std::nullopt;
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  Var var = op->var;
  ffi::Any previous_remap = VarRemapGet(var);
  // The ordinary Bind path rewrites the RHS before defining the Var and propagates its type.
  Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  Var bound_var = stmt.as<BindNode>()->var;
  if (!bound_var.same_as(GetVarRemap(var).value_or(var))) SetVarRemap(var, bound_var);
  VarRemapSet(var, previous_remap);
  return stmt;
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const IfNode* op, InplaceMode inplace_mode) {
  // Each branch gets its own scope so Bind remaps in one branch
  // do not leak into the other.
  auto condition_result = Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  Stmt then_case = scope_.WithNewScope([&]() -> Stmt {
    return Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
  });
  ffi::Optional<SeqStmt> else_case;
  if (op->else_case) {
    else_case = scope_.WithNewScope([&]() -> SeqStmt {
      return Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
    });
  }
  if (condition_unchanged && then_case.same_as(op->then_case) && else_case.same_as(op->else_case)) {
    return ffi::Unchanged();
  }
  return If(condition, then_case, else_case);
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const ForNode* op, InplaceMode inplace_mode) {
  const Var& v = op->loop_var;
  if (defined_.count(v.get())) {
    return scope_.WithNewScope([&]() -> Stmt {
      Var new_var = MakeNewVar(v);
      SetVarRemap(v, new_var);
      Stmt stmt =
          StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      auto n = ffi::make_object<ForNode>(*stmt.as<ForNode>());
      n->loop_var = new_var.as_or_throw<PrimVar>();
      return For(n);
    });
  } else {
    defined_.insert(v.get());
    return scope_.WithNewScope([&]() -> Stmt {
      SetVarRemap(v, v);
      return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    });
  }
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const WhileNode* op, InplaceMode inplace_mode) {
  return scope_.WithNewScope([&]() -> Stmt {
    return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  });
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) {
  auto attrs = Mutate(op->attrs, InplaceMode::kDisallow)
                   .as_or_throw<UnchangedOr<DictAttrs>>()
                   .ValueOrUnchanged(op->attrs);
  auto args = Mutate(op->args, InplaceMode::kDisallow)
                  .as_or_throw<UnchangedOr<ffi::Array<Expr>>>()
                  .ValueOrUnchanged(op->args);
  ffi::Array<Var> params;
  Stmt body = scope_.WithNewScope([&]() -> Stmt {
    for (const Var& var : op->body_params) params.push_back(DefineVar(var));
    return Mutate(op->body, InplaceMode::kDisallow).ValueOrUnchanged(op->body);
  });
  ffi::Array<Var> results;
  for (const Var& var : op->result_vars) results.push_back(DefineVar(var));
  return RegionStmt(op->op, std::move(args), std::move(params), std::move(attrs), std::move(body),
                    std::move(results), op->loc);
}

Var IRConvertSSA::MakeNewVar(const Var& old_var) {
  return Var(old_var->name, old_var->ty, old_var->loc);
}

void IRConvertSSA::SetVarRemap(const Var& old_var, const Var& new_var) {
  scope_.Current().insert_or_assign(old_var.get(), new_var);
}

IRModule IRConvertSSA::VisitIRModule(IRModule mod) {
  ffi::Map<GlobalVar, BaseFunc> functions;
  bool made_change = false;
  for (auto [gvar, base_func] : mod->functions) {
    if (auto* ptr = base_func.as<tirx::FunctionNode>()) {
      auto updated = VisitFunction(ffi::GetRef<tirx::Function>(ptr));
      if (!updated.same_as(base_func)) {
        made_change = true;
        base_func = updated;
      }
    }
    functions.Set(gvar, base_func);
  }
  if (made_change) {
    mod.CopyOnWrite()->functions = std::move(functions);
  }
  return mod;
}
}  // namespace

Stmt ConvertSSA(Stmt stmt) {
  return ffi::make_object<IRConvertSSA>()->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
}

namespace transform {
Pass ConvertSSA() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    return ffi::make_object<tirx::IRConvertSSA>()->VisitIRModule(std::move(mod));
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tirx.ConvertSSA");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.ConvertSSA", ConvertSSA);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
