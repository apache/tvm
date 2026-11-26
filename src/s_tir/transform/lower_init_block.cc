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
 * Lower block init stmt into branch stmt
 * \file lower_reduction.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>

#include <optional>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

namespace {

/*! \brief Substitute variables according to a remapping. */
class VarSubstituter : public StmtExprMutator {
 public:
  explicit VarSubstituter(const ffi::Map<Var, PrimExpr>& substitutions) {
    for (const auto& [var, value] : substitutions) {
      VarRemapSet(var, value);
    }
  }

  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
};

/*! \brief Check whether a loop executes zero iterations for a zero extent. */
bool IsStaticallyNonEmpty(const ForObj* loop) {
  std::optional<IntImm> begin = loop->begin.as<IntImm>();
  std::optional<IntImm> end = loop->end.as<IntImm>();
  return begin.has_value() && end.has_value() && end.value()->value > begin.value()->value;
}

}  // namespace

class InitBlockLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

 private:
  /*!
   * \brief Location at which a lowered init statement was found.
   *
   * The init statement is lowered into `if (reduce_var == begin) init`, which
   * is skipped when the reduction loop has zero extent.  To keep the zero
   * extent case correct we record the enclosing blocks and loops so that a
   * zero-extent fallback can be emitted outside of the reduction loops.
   */
  struct InitSite {
    // The block realize that owns the init statement.
    const SBlockRealizeObj* realize;
    // The lowered init statement.
    const IfObj* init;
    // Loops between the current loop and the init statement, outermost first.
    std::vector<const ForObj*> loops;
  };

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    if (!block->init.has_value()) {
      return StmtExprMutator::Mutate_(block, inplace_mode);
    }
    Stmt init = DoLowering(block->init.value(), block->iter_vars);
    if (const auto* init_if = init.as<IfObj>()) {
      init_stmts_.insert(init_if);
    }
    Stmt body = Mutate(block->body, inplace_mode).ValueOrUnchanged(block->body);
    body = SeqStmt({init, body});
    if (inplace_mode == InplaceMode::kAllow) {
      auto* writable = const_cast<SBlockObj*>(block);
      writable->init = std::nullopt;
      writable->body = std::move(body);
      return ffi::Unchanged();
    } else {
      auto copy = ffi::make_object<SBlockObj>(*block);
      copy->init = std::nullopt;
      copy->body = std::move(body);
      return SBlock(std::move(copy));
    }
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    Stmt stmt =
        StmtExprMutator::Mutate_(loop, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
    const auto* new_loop = stmt.as<ForObj>();
    TVM_FFI_ICHECK(new_loop != nullptr);

    // A statically non-empty loop never needs the zero-extent fallback, so the
    // common case keeps generating exactly the same code as before.
    if (!new_loop->annotations.count(tvm::s_tir::attr::kThreadBinding.name) &&
        !IsStaticallyNonEmpty(new_loop)) {
      std::vector<const ForObj*> loops{new_loop};
      if (auto site = FindInitSite(new_loop->body, &loops)) {
        std::optional<Stmt> fallback = MakeZeroExtentFallback(*site, new_loop);
        if (fallback.has_value()) {
          Stmt seq = SeqStmt({fallback.value(), stmt});
          if (const auto* fallback_if = fallback.value().as<IfObj>()) {
            zero_extent_fallbacks_.insert(fallback_if);
          }
          return seq;
        }
      }
    }
    return stmt;
  }

  /*! \brief Find the lowered init statement reachable from \p stmt along a chain of loops. */
  std::optional<InitSite> FindInitSite(const Stmt& stmt, std::vector<const ForObj*>* loops) {
    if (const auto* realize = stmt.as<SBlockRealizeObj>()) {
      std::optional<IntImm> predicate = realize->predicate.as<IntImm>();
      if (!predicate.has_value() || !predicate.value()->value) {
        return std::nullopt;
      }
      const SBlockObj* block = realize->block.get();
      if (block->body.size() == 0) {
        return std::nullopt;
      }
      const auto* init = block->body[0].as<IfObj>();
      if (init == nullptr || !init_stmts_.count(init)) {
        return std::nullopt;
      }
      return InitSite{realize, init, *loops};
    }
    if (const auto* seq = stmt.as<SeqStmtObj>()) {
      // Skip zero-extent fallbacks inserted for inner loops and require that
      // the loop body contains a single meaningful statement.
      const Stmt* body = nullptr;
      for (const Stmt& item : seq->seq) {
        if (const auto* if_node = item.as<IfObj>();
            if_node != nullptr && zero_extent_fallbacks_.count(if_node)) {
          continue;
        }
        if (body != nullptr) {
          return std::nullopt;
        }
        body = &item;
      }
      if (body == nullptr) {
        return std::nullopt;
      }
      return FindInitSite(*body, loops);
    }
    if (const auto* inner = stmt.as<ForObj>()) {
      loops->push_back(inner);
      return FindInitSite(inner->body, loops);
    }
    return std::nullopt;
  }

  /*! \brief Build the init statement executed when \p loop has zero iterations. */
  std::optional<Stmt> MakeZeroExtentFallback(const InitSite& site, const ForObj* loop) {
    // The fallback is emitted outside of `loop`, which therefore has to be a
    // reduction loop of the init-owning block.
    if (!IsReductionLoop(site.realize, loop)) {
      return std::nullopt;
    }
    // Require every loop between `loop` and the init to be a reduction loop of
    // the init-owning block.  Moving the init across a spatial loop would only
    // initialize a single slice of the output.
    for (const ForObj* path_loop : site.loops) {
      if (!IsReductionLoop(site.realize, path_loop)) {
        return std::nullopt;
      }
    }

    // Replace the block iter vars with their values.  The fallback is emitted
    // outside of the block realize, so they are no longer in scope.
    ffi::Map<Var, PrimExpr> substitutions;
    for (size_t i = 0; i < site.realize->block->iter_vars.size(); ++i) {
      const IterVar& iter_var = site.realize->block->iter_vars[i];
      const PrimExpr& value = site.realize->iter_values[i];
      substitutions.Set(iter_var->var, value);
    }
    Stmt init = ffi::make_object<VarSubstituter>(substitutions)
                    ->Mutate(static_cast<const Stmt&>(site.init->then_case))
                    .ValueOrUnchanged(site.init->then_case);

    // The init must be independent of the reduction loops being crossed.
    // Otherwise evaluating it outside of them would change its meaning.
    std::unordered_set<const VarObj*> crossed_loops{loop->var.get()};
    for (const ForObj* path_loop : site.loops) {
      crossed_loops.insert(path_loop->var.get());
    }
    for (const Var& var : UndefinedVars(init)) {
      if (crossed_loops.count(var.get())) {
        return std::nullopt;
      }
    }

    PrimExpr is_zero_extent = equal(loop->begin, loop->end);
    return If(is_zero_extent, init.as_or_throw<SeqStmt>());
  }

  /*! \brief Check whether \p loop iterates over a reduction axis of \p realize. */
  static bool IsReductionLoop(const SBlockRealizeObj* realize, const ForObj* loop) {
    const VarObj* loop_var = loop->var.get();
    for (size_t i = 0; i < realize->block->iter_vars.size(); ++i) {
      if (realize->block->iter_vars[i]->iter_type != IterVarType::kCommReduce) {
        continue;
      }
      const auto* value = realize->iter_values[i].as<VarObj>();
      if (value == loop_var) {
        return true;
      }
    }
    return false;
  }

  static Stmt DoLowering(const Stmt& init, const ffi::Array<IterVar>& iter_vars) {
    std::vector<PrimExpr> conditions;
    for (const IterVar& var : iter_vars) {
      if (var->iter_type == IterVarType::kCommReduce) {
        conditions.push_back(equal(var->var, var->dom.value()->begin));
      }
    }
    // Handle the case where there is no condition
    if (conditions.empty()) {
      return init;
    }
    // Concat the conditions with logical and (&&)
    PrimExpr cond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i) {
      cond = logical_and(cond, conditions[i]);
    }
    return If(cond, init.as_or_throw<SeqStmt>());
  }

  // Init statements produced by this pass, used to identify them in the IR.
  std::unordered_set<const IfObj*> init_stmts_;
  // Zero-extent fallbacks produced by this pass.
  std::unordered_set<const IfObj*> zero_extent_fallbacks_;
};

namespace transform {

Pass LowerInitBlock() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    fptr->body = ffi::make_object<InitBlockLower>()
                     ->Mutate(fptr->body, InplaceMode::kAllow)
                     .ValueOrUnchanged(std::move(fptr->body));
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerInitBlock");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerInitBlock", LowerInitBlock);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
