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
 * \file tvm/lowered_func.h
 * \brief Information about a lowered TVM function.
 *  This data structure is final step toward codegen.
 */
#ifndef TVM_LOWERED_FUNC_H_
#define TVM_LOWERED_FUNC_H_

#include <string>

#include "base.h"
#include "expr.h"
#include "tensor.h"
#include "tvm/node/container.h"

namespace tvm {

// Internal node container of lowered function.
class LoweredFuncNode;

/*!
 * \brief LoweredFunc represents function after lowering.
 *  This is the final IR representation before codegen.
 */
class LoweredFunc : public ir::FunctionRef {
 public:
  LoweredFunc() {}
  explicit LoweredFunc(NodePtr<Node> n) : FunctionRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const LoweredFuncNode* operator->() const;
  /*! \brief specify container node */
  using ContainerType = LoweredFuncNode;
};

/*! \brief specific type of lowered function */
enum LoweredFuncType : int {
  /*! \brief Function that can mix device and host calls */
  kMixedFunc = 0,
  /*! \brief Only contains host code */
  kHostFunc = 1,
  /*! \brief Only contains device code */
  kDeviceFunc = 2
};

/*! \brief Node container of LoweredFunc */
class LoweredFuncNode : public ir::FunctionBaseNode {
 public:
  /*! \brief The name of the function */
  std::string name;
  /*!
   * \brief The arguments of the function
   *  This function can only take pod type(int, float) and void* as arguments.
   */
  Array<Var> args;
  /*!
   * \brief The IterVar axis of threads
   *  Each axis need host function to specify a size.
   * \note Calling convention into LoweredFunc
   *
   * Assume we have a LoweredFunc f, a call into f
   *   Call(f, arg1, arg2, ..., arg_n,
   *        size_axis_1, size_axis_2, ... size_axis_m)
   *
   * Here n = len(args), m = len(thread_axis)
   *
   * The CodeGen should take this and translate this call
   * to corresponding API specific kernel launchs or function calls.
   */
  Array<IterVar> thread_axis;
  /*!
   * \brief The hint data type of Var handles defined in LetStmt
   *  Can be used as hint when generating type signiture.
   *  The creation rule is given by
   *  handle_data_type[var_handle] = make_const(the_type, 0);
   *
   * \note Expr is used instead Type, because Type cannot be hold by Map.
   *  constant Expr of given type is used.
   */
  Map<Var, Expr> handle_data_type;
  /*! \brief The type of the function */
  LoweredFuncType func_type{kMixedFunc};
  /*! \brief Whether this function is packed function */
  bool is_packed_func{true};
  /*!
   * \brief Whether function ensures that argument pointers do not alias.
   *  This corresponds to restrict keyword in C.
   */
  bool is_restricted{true};
  /*! \brief The body statment of the function */
  Stmt body;
  /*! \return name of the operation */
  const std::string& func_name() const final {
    return name;
  }
  // there is no return value, but return 1
  // to enable Call into this function.
  int num_outputs() const final {
    return 1;
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("thread_axis", &thread_axis);
    v->Visit("handle_data_type", &handle_data_type);
    v->Visit("func_type", &func_type);
    v->Visit("is_packed_func", &is_packed_func);
    v->Visit("is_restricted", &is_restricted);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "LoweredFunc";
  TVM_DECLARE_NODE_TYPE_INFO(LoweredFuncNode, Node);
};

// Implementations of inline functions
inline const LoweredFuncNode* LoweredFunc::operator->() const {
  return static_cast<const LoweredFuncNode*>(node_.get());
}

}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::LoweredFunc> {
  std::size_t operator()(const ::tvm::LoweredFunc& k) const {
    return k.hash();
  }
};
}

#endif  // TVM_LOWERED_FUNC_H_
