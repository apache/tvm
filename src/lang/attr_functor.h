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
 *  Copyright (c) 2018 by Contributors
 * \file attr_functor.h
 * \brief A way to define arbitrary function signature
 *        with dispatch on common attributes.
 *
 * Common attributes include:
 *  - int, float, str constants
 *  - array of attributes
 *  - map of attributes
 */
#ifndef TVM_LANG_ATTR_FUNCTOR_H_
#define TVM_LANG_ATTR_FUNCTOR_H_

#include <utility>

namespace tvm {

template <typename FType>
class AttrFunctor;

#define ATTR_FUNCTOR_DEFAULT                                        \
  { return VisitAttrDefault_(op, std::forward<Args>(args)...); }


#define ATTR_FUNCTOR_DISPATCH(OP)                                       \
  vtable.template set_dispatch<OP>(                                     \
      [](const NodeRef& n, TSelf* self, Args... args) {                 \
        return self->VisitAttr_(static_cast<const OP*>(n.node_.get()),  \
                                std::forward<Args>(args)...);           \
      });                                                               \

// A functor for common attribute information.
template <typename R, typename... Args>
class AttrFunctor<R(const NodeRef& n, Args...)> {
 private:
  using TSelf = AttrFunctor<R(const NodeRef& n, Args...)>;
  using FType = tvm::IRFunctor<R(const NodeRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitAttr(const NodeRef& n, Args... args) {
    static FType vtable = InitVTable();
    if (vtable.can_dispatch(n)) {
      return vtable(n, this, std::forward<Args>(args)...);
    } else {
      return VisitAttrDefault_(n.get(), std::forward<Args>(args)...);
    }
  }
  virtual R VisitAttrDefault_(const Node* node, Args... args) = 0;
  virtual R VisitAttr_(const ArrayNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const StrMapNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::IntImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::UIntImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::FloatImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::StringImm* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  // deep comparison of symbolic integer expressions.
  virtual R VisitAttr_(const Variable* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Add* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Sub* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Mul* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Div* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Mod* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::FloorDiv* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::FloorMod* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Min* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Max* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::GE* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::GT* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::LT* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::LE* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::EQ* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::NE* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::And* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Or* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Not* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Cast* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Call* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const ir::Select* op, Args... args) ATTR_FUNCTOR_DEFAULT;

 private:
  // initialize the vtable.
  static FType InitVTable() {
    using namespace ir;
    FType vtable;
    // Set dispatch
    ATTR_FUNCTOR_DISPATCH(StrMapNode);
    ATTR_FUNCTOR_DISPATCH(ArrayNode);
    ATTR_FUNCTOR_DISPATCH(IntImm);
    ATTR_FUNCTOR_DISPATCH(UIntImm);
    ATTR_FUNCTOR_DISPATCH(FloatImm);
    ATTR_FUNCTOR_DISPATCH(StringImm);
    ATTR_FUNCTOR_DISPATCH(Variable);
    ATTR_FUNCTOR_DISPATCH(Add);
    ATTR_FUNCTOR_DISPATCH(Sub);
    ATTR_FUNCTOR_DISPATCH(Mul);
    ATTR_FUNCTOR_DISPATCH(Div);
    ATTR_FUNCTOR_DISPATCH(Mod);
    ATTR_FUNCTOR_DISPATCH(FloorDiv);
    ATTR_FUNCTOR_DISPATCH(FloorMod);
    ATTR_FUNCTOR_DISPATCH(Min);
    ATTR_FUNCTOR_DISPATCH(Max);
    ATTR_FUNCTOR_DISPATCH(GE);
    ATTR_FUNCTOR_DISPATCH(GT);
    ATTR_FUNCTOR_DISPATCH(LE);
    ATTR_FUNCTOR_DISPATCH(LT);
    ATTR_FUNCTOR_DISPATCH(EQ);
    ATTR_FUNCTOR_DISPATCH(NE);
    ATTR_FUNCTOR_DISPATCH(And);
    ATTR_FUNCTOR_DISPATCH(Or);
    ATTR_FUNCTOR_DISPATCH(Not);
    ATTR_FUNCTOR_DISPATCH(Cast);
    ATTR_FUNCTOR_DISPATCH(Call);
    ATTR_FUNCTOR_DISPATCH(Select);
    return vtable;
  }
};

class AttrsEqualHandler :
      protected AttrFunctor<bool(const NodeRef&, const NodeRef&)> {
 public:
  /*!
   * \brief Check if lhs equals rhs
   * \param lhs The left operand.
   * \param rhs The right operand.
   */
  bool Equal(const NodeRef& lhs, const NodeRef& rhs);

 protected:
  bool VisitAttrDefault_(const Node* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ArrayNode* lhs, const NodeRef& other) final;
  bool VisitAttr_(const StrMapNode* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::IntImm* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::UIntImm* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::FloatImm* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::StringImm* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Add* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Sub* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Mul* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Div* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Mod* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::FloorDiv* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::FloorMod* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Min* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Max* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::GE* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::GT* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::LT* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::LE* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::EQ* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::NE* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::And* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Or* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Not* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Cast* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Call* lhs, const NodeRef& other) final;
  bool VisitAttr_(const ir::Select* lhs, const NodeRef& other) final;
};

class AttrsHashHandler :
      protected AttrFunctor<size_t(const NodeRef&)> {
 public:
  /*!
   * \brief Get hash value of node
   * \param node The node to be hashed.
   */
  size_t Hash(const NodeRef& node) {
    if (!node.defined()) return 0;
    return this->VisitAttr(node);
  }

 protected:
  size_t VisitAttrDefault_(const Node* lhs) final;
  size_t VisitAttr_(const ir::IntImm* lhs) final;
  size_t VisitAttr_(const ir::UIntImm* lhs) final;
  size_t VisitAttr_(const ir::FloatImm* lhs) final;
  size_t VisitAttr_(const ir::StringImm* lhs) final;
  size_t VisitAttr_(const ArrayNode* lhs) final;
  size_t VisitAttr_(const StrMapNode* lhs) final;
  size_t VisitAttr_(const ir::Add* op) final;
  size_t VisitAttr_(const ir::Sub* op) final;
  size_t VisitAttr_(const ir::Mul* op) final;
  size_t VisitAttr_(const ir::Div* op) final;
  size_t VisitAttr_(const ir::Mod* op) final;
  size_t VisitAttr_(const ir::FloorDiv* op) final;
  size_t VisitAttr_(const ir::FloorMod* op) final;
  size_t VisitAttr_(const ir::Min* op) final;
  size_t VisitAttr_(const ir::Max* op) final;
  size_t VisitAttr_(const ir::GE* op) final;
  size_t VisitAttr_(const ir::GT* op) final;
  size_t VisitAttr_(const ir::LE* op) final;
  size_t VisitAttr_(const ir::LT* op) final;
  size_t VisitAttr_(const ir::EQ* op) final;
  size_t VisitAttr_(const ir::NE* op) final;
  size_t VisitAttr_(const ir::And* op) final;
  size_t VisitAttr_(const ir::Or* op) final;
  size_t VisitAttr_(const ir::Not* op) final;
  size_t VisitAttr_(const ir::Cast* op) final;
  size_t VisitAttr_(const ir::Call* op) final;
  size_t VisitAttr_(const ir::Select* op) final;
  /*!
   * \brief alias of dmlc::HashCombine
   * \param lhs The first hash value.
   * \param rhs The second hash value.
   */
  static size_t Combine(size_t lhs, size_t rhs) {
    return dmlc::HashCombine(lhs, rhs);
  }
};
}  // namespace tvm
#endif  // TVM_LANG_ATTR_FUNCTOR_H_
