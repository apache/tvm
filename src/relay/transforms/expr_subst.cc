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
 * \file expr_subst.h
 * \brief Utility functions for substituting expressions.
 */

#include "./expr_subst.h"

#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

class ExprSubstituter : public ExprMutator {
 public:
  explicit ExprSubstituter(std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> subst_map)
      : subst_map_(subst_map) {}

  Expr VisitExpr(const Expr& expr) final {
    auto it = subst_map_.find(expr);
    if (it != subst_map_.end()) {
      return ExprMutator::VisitExpr((*it).second);
    }
    return ExprMutator::VisitExpr(expr);
  }

 private:
  tvm::Map<Expr, Expr> subst_map_;
};

Expr ExprSubst(const Expr& expr,
               std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> subst_map) {
  return ExprSubstituter(std::move(subst_map)).Mutate(expr);
}

}  // namespace relay
}  // namespace tvm
