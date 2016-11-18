/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_pass.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>

namespace tvm {
namespace ir {
namespace {
// visitor to implement apply
class IRSubstitute : public IRMutator {
 public:
  Expr Mutate(Expr expr) final {
    const IRNode* v = expr.get();
    if (v != nullptr) {
      auto it = replacements_.find(v);
      if (it != replacements_.end()) {
        return it->second;
      }
    }
    return IRMutator::Mutate(expr);
  }
  explicit IRSubstitute(const std::unordered_map<const IRNode*, Expr>& replacements)
      : replacements_(replacements) {}

 private:
  const std::unordered_map<const IRNode*, Expr>& replacements_;
};
}  // namespace

Expr Substitute(const std::unordered_map<const IRNode*, Expr>& replacements, Expr expr) {
  return IRSubstitute(replacements).Mutate(expr);
}

}  // namespace ir
}  // namespace tvm
