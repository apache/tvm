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
 *
 * \file tvm/relay/_transforms/pass_util.h
 * \brief Utilities for writing passes
 */
#ifndef TVM_RELAY_TRANSFORMS_PASS_UTIL_H_
#define TVM_RELAY_TRANSFORMS_PASS_UTIL_H_

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

#include <memory>
#include <unordered_map>

namespace tvm {
namespace relay {

/*!
 * \brief Check if expr is positive constant.
 * \param expr The expression to be checked.
 * \return Whether all elements of expr is positive constant.
 */
bool IsAllPositiveConstant(const Expr& expr);

/*!
 * \brief Substitute var with subst.
 * \param type The type to be substituted.
 * \param tvar The type variable to be substituted.
 * \param subst The target of substitution.
 * \return The substituted result.
 */
Type TypeSubst(const Type& type, const TypeVar& tvar, const Type& subst);

/*!
 * \brief Substitute var with subst.
 * \param expr The expr to be substituted.
 * \param tvar The type variable to be substituted.
 * \param subst The target of substitution.
 * \return The substituted result.
 */
Expr TypeSubst(const Expr& expr, const TypeVar& tvar, const Type& subst);

/*!
 * \brief Substitute type vars in type.
 * \param type The type to be substituted.
 * \param subst_map The map of substitution.
 * \return The substituted result.
 */
Type TypeSubst(const Type& type, const tvm::Map<TypeVar, Type>& subst_map);

/*!
 * \brief Substitute type vars in type.
 * \param expr The expr to be substituted.
 * \param subst_map The map of substitution.
 * \return The substituted result.
 */
Expr TypeSubst(const Expr& expr, const tvm::Map<TypeVar, Type>& subst_map);

/*!
 * \brief Check if type is dynamic.
 * \param ty The type to be checked.
 * \return Whether the type is dynamic.
 */
bool IsDynamic(const Type& ty);

/*!
 * \brief Check if call is data dependant.
 * \param call The call to be checked.
 * \return Whether the call is data dependant.
 */
bool IsDataDependant(const CallNode* call);

/*!
 * \brief Make arbitrary transformation preserve the out most function.
 * \param func The transformation.
 * \param e The expression
 * \return the transformed expression. If e is a function the return is also a function.
 */
inline Expr TransformF(const std::function<Expr(const Expr&)>& func, const Expr& e) {
  if (const FunctionNode* f = e.as<FunctionNode>()) {
    return Function(f->params, func(f->body), f->ret_type, f->type_params, f->attrs);
  } else {
    return func(e);
  }
}

/*!
 * \brief Decide whether the expression atomic or not?
 * \param e the expression
 * \return
 *   is it atomic?
 *   if so, the compute cost of the expression is bounded so it can be copy without graph mode.
 */
inline bool IsAtomic(const Expr& e) {
  return e.as<VarNode>() || e.as<OpNode>() || e.as<ConstructorNode>() || e.as<GlobalVarNode>();
}

/*!
 * \brief Cache the compiler_begin annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_begin op
 */
inline const Op& CompilerBeginOp() {
  static auto op = Op::Get("annotation.compiler_begin");
  return op;
}

/*!
 * \brief Cache the compiler_end annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_end op
 */
inline const Op& CompilerEndOp() {
  static auto op = Op::Get("annotation.compiler_end");
  return op;
}

template <typename ConditionObjectPtr>
struct TreeNode {
  typedef std::shared_ptr<TreeNode<ConditionObjectPtr>> pointer;
  virtual ~TreeNode() {}
};

template <typename ConditionObjectPtr>
struct TreeLeafNode : TreeNode<ConditionObjectPtr> {
  using TreeObjectPtr = typename TreeNode<ConditionObjectPtr>::pointer;

  Expr body;

  explicit TreeLeafNode(Expr body) : body(body) {}

  static TreeObjectPtr Make(Expr body) { return std::make_shared<TreeLeafNode>(body); }

  ~TreeLeafNode() {}
};

template <typename ConditionObjectPtr>
struct TreeLeafFatalNode : TreeNode<ConditionObjectPtr> {
  using TreeObjectPtr = typename TreeNode<ConditionObjectPtr>::pointer;

  TreeLeafFatalNode() = default;

  static TreeObjectPtr Make() { return std::make_shared<TreeLeafFatalNode>(); }

  ~TreeLeafFatalNode() {}
};

template <typename ConditionObjectPtr>
struct TreeBranchNode : TreeNode<ConditionObjectPtr> {
  using TreeObjectPtr = typename TreeNode<ConditionObjectPtr>::pointer;

  ConditionObjectPtr cond;
  TreeObjectPtr then_branch;
  TreeObjectPtr else_branch;

  TreeBranchNode(ConditionObjectPtr cond, TreeObjectPtr then_branch, TreeObjectPtr else_branch)
      : cond(cond), then_branch(then_branch), else_branch(else_branch) {}

  static TreeObjectPtr Make(ConditionObjectPtr cond, TreeObjectPtr then_branch,
                            TreeObjectPtr else_branch) {
    return std::make_shared<TreeBranchNode>(cond, then_branch, else_branch);
  }

  ~TreeBranchNode() {}
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_PASS_UTIL_H_
