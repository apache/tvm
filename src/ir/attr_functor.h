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
 * \file attr_functor.h
 * \brief A way to define arbitrary function signature
 *        with dispatch on common attributes.
 *
 * Common attributes include:
 *  - int, float, str constants
 *  - array of attributes
 *  - map of attributes
 */
#ifndef TVM_IR_ATTR_FUNCTOR_H_
#define TVM_IR_ATTR_FUNCTOR_H_

#include <tvm/node/functor.h>
#include <tvm/tir/expr.h>

#include <utility>

namespace tvm {

template <typename FType>
class AttrFunctor;

#define ATTR_FUNCTOR_DEFAULT \
  { return VisitAttrDefault_(op, std::forward<Args>(args)...); }

#define ATTR_FUNCTOR_DISPATCH(OP)                                                          \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitAttr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

// A functor for common attribute information.
template <typename R, typename... Args>
class AttrFunctor<R(const ObjectRef& n, Args...)> {
 private:
  using TSelf = AttrFunctor<R(const ObjectRef& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~AttrFunctor() {}
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitAttr(const ObjectRef& n, Args... args) {
    static FType vtable = InitVTable();
    if (vtable.can_dispatch(n)) {
      return vtable(n, this, std::forward<Args>(args)...);
    } else {
      return VisitAttrDefault_(n.get(), std::forward<Args>(args)...);
    }
  }
  virtual R VisitAttrDefault_(const Object* node, Args... args) = 0;
  virtual R VisitAttr_(const ArrayNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::IntImmNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::FloatImmNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::StringImmNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  // deep comparison of symbolic integer expressions.
  virtual R VisitAttr_(const tir::VarNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::SizeVarNode* op, Args... args) {
    return VisitAttr_(static_cast<const tir::VarNode*>(op), std::forward<Args>(args)...);
  }
  virtual R VisitAttr_(const tir::AddNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::SubNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::MulNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::DivNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::ModNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::FloorDivNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::FloorModNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::MinNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::MaxNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::GENode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::GTNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::LTNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::LENode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::EQNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::NENode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::AndNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::OrNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::NotNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::CastNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::CallNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;
  virtual R VisitAttr_(const tir::SelectNode* op, Args... args) ATTR_FUNCTOR_DEFAULT;

 private:
  // initialize the vtable.
  static FType InitVTable() {
    using namespace tir;
    FType vtable;
    // Set dispatch
    ATTR_FUNCTOR_DISPATCH(ArrayNode);
    ATTR_FUNCTOR_DISPATCH(IntImmNode);
    ATTR_FUNCTOR_DISPATCH(FloatImmNode);
    ATTR_FUNCTOR_DISPATCH(StringImmNode);
    ATTR_FUNCTOR_DISPATCH(VarNode);
    ATTR_FUNCTOR_DISPATCH(SizeVarNode);
    ATTR_FUNCTOR_DISPATCH(AddNode);
    ATTR_FUNCTOR_DISPATCH(SubNode);
    ATTR_FUNCTOR_DISPATCH(MulNode);
    ATTR_FUNCTOR_DISPATCH(DivNode);
    ATTR_FUNCTOR_DISPATCH(ModNode);
    ATTR_FUNCTOR_DISPATCH(FloorDivNode);
    ATTR_FUNCTOR_DISPATCH(FloorModNode);
    ATTR_FUNCTOR_DISPATCH(MinNode);
    ATTR_FUNCTOR_DISPATCH(MaxNode);
    ATTR_FUNCTOR_DISPATCH(GENode);
    ATTR_FUNCTOR_DISPATCH(GTNode);
    ATTR_FUNCTOR_DISPATCH(LENode);
    ATTR_FUNCTOR_DISPATCH(LTNode);
    ATTR_FUNCTOR_DISPATCH(EQNode);
    ATTR_FUNCTOR_DISPATCH(NENode);
    ATTR_FUNCTOR_DISPATCH(AndNode);
    ATTR_FUNCTOR_DISPATCH(OrNode);
    ATTR_FUNCTOR_DISPATCH(NotNode);
    ATTR_FUNCTOR_DISPATCH(CastNode);
    ATTR_FUNCTOR_DISPATCH(CallNode);
    ATTR_FUNCTOR_DISPATCH(SelectNode);
    return vtable;
  }
};

}  // namespace tvm
#endif  // TVM_IR_ATTR_FUNCTOR_H_
