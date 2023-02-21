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
 * \file src/relax/transform/utils.h
 * \brief Additional utility classes and functions for working with the Relax IR.
 */
#ifndef TVM_RELAX_TRANSFORM_UTILS_H_
#define TVM_RELAX_TRANSFORM_UTILS_H_

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <string>
#include <unordered_map>

#include "../../relay/analysis/graph_partitioner.h"

namespace tvm {
namespace relax {

/*!
 * \brief A simple wrapper around ExprFunctor for a single argument case.
 *  The result of visit is memoized.
 */
template <typename OutputType>
class MemoizedExprTranslator : public ::tvm::relax::ExprFunctor<OutputType(const Expr&)> {
  using BaseFunctor = ::tvm::relax::ExprFunctor<OutputType(const Expr&)>;

 public:
  /*! \brief virtual destructor */
  virtual ~MemoizedExprTranslator() {}

  /*!
   * \brief The memoized call.
   * \param n The expression node.
   * \return The result of the call
   */
  virtual OutputType VisitExpr(const Expr& n) {
    ICHECK(n.defined());
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return it->second;
    }
    auto res = BaseFunctor::VisitExpr(n);
    memo_[n] = res;
    return res;
  }

  virtual OutputType VisitExpr_(const VarNode* vn) {
    ICHECK(memo_.count(GetRef<Expr>(vn)));
    return memo_[GetRef<Expr>(vn)];
  }

  virtual OutputType VisitBinding_(const VarBindingNode* binding) {
    ICHECK_EQ(memo_.count(binding->var), 0);
    auto v = VisitExpr(binding->value);
    memo_[binding->var] = v;
    return v;
  }

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, OutputType, ObjectPtrHash, ObjectPtrEqual> memo_;
};

/*!
 * \brief Remove unused global relax functions in an IRModule.
 * \param mod The target module
 * \param entry_functions list of entry functions
 * \return The updated module.
 */
TVM_DLL IRModule RemoveUnusedFunctions(IRModule mod, Array<runtime::String> entry_funcs);

/*!
 * \brief Get the external symbol of the Relax function name.
 *
 * \param func The provided function.
 * \return An external symbol.
 */
inline std::string GetExtSymbol(const Function& func) {
  const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(name_node.defined()) << "Fail to retrieve external symbol.";
  return std::string(name_node.value());
}

/*!
 * \brief Fuse ops or functions according to the given partition, and grouped them into a new
 * function.
 *
 * \param mod The input module.
 * \param partition A mapping from a subexpression to the containing group.
 * \param lift_constants Whether or not to lift bound constants to parameters of the
 * grouped function.
 * \return A new module containing grouped functions.
 */
IRModule MakeGroupedFunctions(
    IRModule mod,
    const std::unordered_map<const Object*, relay::GraphPartitioner::Group*>& partition,
    bool lift_constants = true);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_UTILS_H_
