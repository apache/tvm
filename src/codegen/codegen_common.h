/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_common.h
 * \brief Common utility for codegen.
 */
#ifndef TVM_CODEGEN_CODEGEN_COMMON_H_
#define TVM_CODEGEN_CODEGEN_COMMON_H_

#include <tvm/arithmetic.h>
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Visit AssertStmt recursively, update align_map from condition.
 * \param op The AssertStmt
 * \param align_map The alignmap
 * \param fvisit The recursive visitor
 * \tparam FVisit the recursive visitor
 */
template<typename FVisit>
inline void VisitAssert(
    const ir::AssertStmt* op,
    std::unordered_map<const Variable*, arith::ModularEntry>* align_map,
    FVisit fvisit) {
  using namespace ir;
  auto& align_map_ = *align_map;
  // Detect useful invariant pattern and use them to visit child.
  // Pattern: Var % const  == 0
  // TODO(tqchen) merge these pattern to a generic scope info visitor.
  if (const EQ* eq = op->condition.as<EQ>()) {
    const Mod* mod = eq->a.as<Mod>();
    int64_t factor = 0, offset = 0;
    if (mod && arith::GetConst(eq->b, &offset)) {
      const Variable *var = mod->a.as<Variable>();
      if (var && arith::GetConst(mod->b, &factor)) {
        arith::ModularEntry old = align_map_[var];
        if (factor > old.coeff) {
          arith::ModularEntry e;
          e.coeff = static_cast<int>(factor);
          e.base = static_cast<int>(offset);
          // new alignment info,
          align_map_[var] = e;
          fvisit(op->body);
          // restore old info
          align_map_[var] = old;
          return;
        }
      }
    }
  }
  fvisit(op->body);
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_COMMON_H_
