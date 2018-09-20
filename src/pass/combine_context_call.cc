/*!
 *  Copyright (c) 2017 by Contributors
 *  Combine calls into context related function into one.
 *
 * \file combine_context_call.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <map>

namespace tvm {
namespace ir {

// Calculate the statistics of packed function.
// These information are needed during codegen.
class ContextCallCombiner final : public IRMutator {
 public:
  struct CompareExpr {
    bool operator()(const Expr& lhs, const Expr& rhs) const {
      return Compare(lhs, rhs) < 0;
    }
  };

  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->is_intrinsic(intrinsic::tvm_thread_context)) {
      CHECK_EQ(op->args.size(), 1U);
      Expr ctx = op->args[0];
      auto it  = ctx_map_.find(ctx);
      if (it != ctx_map_.end()) {
        return it->second;
      } else {
        CHECK(ctx.type().is_handle());
        std::string name;
        if (const Call* call = ctx.as<Call>()) {
          name = call->name + "_cache";
        } else {
          name = "ctx_cache_";
        }
        Var ctx_var(name, ctx.type());
        ctx_map_[ctx] = ctx_var;
        return ctx_var;
      }
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::coproc_uop_scope) {
      // Map of comparison expression to variable
      std::map<Expr, Var, CompareExpr> temp;
      std::swap(temp, ctx_map_);
      Stmt stmt = IRMutator::Mutate_(op, s);
      std::swap(temp, ctx_map_);
      return BuildContext(temp, stmt);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    if (op->for_type == ForType::Parallel) {
      // Map of comparison expression to variable
      std::map<Expr, Var, CompareExpr> temp;
      std::swap(temp, ctx_map_);
      Stmt stmt = IRMutator::Mutate_(op, s);
      std::swap(temp, ctx_map_);
      return BuildContext(temp, stmt);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Combine(Stmt stmt) {
    return BuildContext(ctx_map_, this->Mutate(stmt));
  }

 private:
  static Stmt BuildContext(const std::map<Expr, Var, CompareExpr>& cmap,
                           Stmt body) {
    for (const auto& kv : cmap) {
      body = LetStmt::make(kv.second, kv.first, body);
    }
    return body;
  }
  // Map of comparison expression to variable
  std::map<Expr, Var, CompareExpr> ctx_map_;
};

LoweredFunc CombineContextCall(LoweredFunc f) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = ContextCallCombiner().Combine(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
