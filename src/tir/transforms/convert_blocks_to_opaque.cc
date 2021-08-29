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
    // Step 1. Update "block vars => binding values" for substitution.
    ICHECK_EQ(block_op->iter_vars.size(), realize->iter_values.size());
    for (int i = 0, n = block_op->iter_vars.size(); i < n; ++i) {
      IterVar block_var = block_op->iter_vars[i];
      PrimExpr v = this->VisitExpr(realize->iter_values[i]);
      var_substitutes_.emplace(block_var->var.get(), v);
    }
    // Step 2. Visit recursively.
    BlockRealize new_realize = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(realize));
    if (!new_realize->iter_values.empty()) {
      new_realize.CopyOnWrite()->iter_values.clear();
    }
    return std::move(new_realize);
  }

  /*! \brief The map from block vars to thier binding values. */
  std::unordered_map<const VarNode*, PrimExpr> var_substitutes_;
};

PrimFunc ConvertBlocksToOpaque(PrimFunc f) {
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = OpaqueBlockConverter::Substitute(f);
  return f;
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
