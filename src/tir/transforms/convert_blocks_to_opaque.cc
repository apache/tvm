/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file convert_block_to_opaque.cc
 * \brief Convert the blocks to opaque blocks which do not have block vars.
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Substitute expr via BlockRealize value bindings and convert each block into opaque
 *        blocks.
 */
class OpaqueBlockConverter : public StmtExprMutator {
 public:
  static Stmt Substitute(const PrimFunc& f) {
    OpaqueBlockConverter substituter;
    return substituter.VisitStmt(f->body);
  }

 private:
  OpaqueBlockConverter() = default;

  PrimExpr VisitExpr_(const VarNode* var) final {
    CHECK(!forbidden_iter_vars_.count(var))
        << "Variable " << var->name_hint << " occurs in the predicate or iter_values of a block, "
        << "but isn't defined until the body of the block";

    auto it = var_substitutes_.find(var);
    if (it != var_substitutes_.end()) {
      return it->second;
    }
    return GetRef<Var>(var);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    ICHECK(!block->init.defined())
        << "Block Init part is not allowed in pass ConvertBlocksToOpaque";
    Block new_block = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
    if (!new_block->iter_vars.empty()) {
      new_block.CopyOnWrite()->iter_vars.clear();
    }
    return std::move(new_block);
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    const auto* block_op = realize->block.get();
    ICHECK(!block_op->init.defined());

    // Step 1. Visit the predicate and iter_values, without any variable bindings
    for (const auto& iter : block_op->iter_vars) forbidden_iter_vars_.insert(iter->var.get());
    PrimExpr predicate = VisitExpr(realize->predicate);
    Array<PrimExpr> iter_values = realize->iter_values;
    iter_values.MutateByApply([this](PrimExpr expr) { return VisitExpr(std::move(expr)); });
    for (const auto& iter : block_op->iter_vars) forbidden_iter_vars_.erase(iter->var.get());

    // Step 2. Update "block vars => binding values" for substitution.
    ICHECK_EQ(block_op->iter_vars.size(), iter_values.size());
    for (int i = 0, n = block_op->iter_vars.size(); i < n; ++i) {
      IterVar block_var = block_op->iter_vars[i];
      PrimExpr v = this->VisitExpr(iter_values[i]);
      var_substitutes_.emplace(block_var->var.get(), v);
    }
    // Step 3. Visit recursively.
    Block new_block = Downcast<Block>(VisitStmt(realize->block));

    // Step 4. Clear the variable bindings
    for (const auto& block_var : block_op->iter_vars) {
      var_substitutes_.erase(block_var->var.get());
    }

    // Step 5. Return
    if (predicate.same_as(realize->predicate) && iter_values.same_as(realize->iter_values) &&
        new_block.same_as(realize->block) && realize->iter_values.size() == 0) {
      return GetRef<BlockRealize>(realize);
    } else {
      return BlockRealize({}, predicate, new_block);
    }
  }

  /*! \brief The map from block vars to their binding values. */
  std::unordered_map<const VarNode*, PrimExpr> var_substitutes_;
  /* \brief Variables that may not occur in the current context */
  std::unordered_set<const VarNode*> forbidden_iter_vars_;
};

PrimFunc ConvertBlocksToOpaque(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = OpaqueBlockConverter::Substitute(f);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass ConvertBlocksToOpaque() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ConvertBlocksToOpaque(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ConvertBlocksToOpaque", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ConvertBlocksToOpaque").set_body_typed(ConvertBlocksToOpaque);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
