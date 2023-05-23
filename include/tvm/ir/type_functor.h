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
 * \file tvm/ir/type_functor.h
 * \brief A way to defined arbitrary function signature with dispatch on types.
 */
#ifndef TVM_IR_TYPE_FUNCTOR_H_
#define TVM_IR_TYPE_FUNCTOR_H_

#include <tvm/ir/tensor_type.h>
#include <tvm/ir/type_relation.h>
#include <tvm/node/functor.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {

template <typename FType>
class TypeFunctor;

// functions to be overriden.
#define TYPE_FUNCTOR_DEFAULT \
  { return VisitTypeDefault_(op, std::forward<Args>(args)...); }

#define TVM_TYPE_FUNCTOR_DISPATCH(OP)                                                      \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitType_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class TypeFunctor<R(const Type& n, Args...)> {
 private:
  using TSelf = TypeFunctor<R(const Type& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~TypeFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Type& n, Args... args) { return VisitType(n, std::forward<Args>(args)...); }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitType(const Type& n, Args... args) {
    ICHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitType_(const TensorTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeVarNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeConstraintNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FuncTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeRelationNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TupleTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const IncompleteTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const RelayRefTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const GlobalTypeVarNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeCallNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TypeDataNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PrimTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PointerTypeNode* op, Args... args) TYPE_FUNCTOR_DEFAULT;
  virtual R VisitTypeDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;  // unreachable, written to stop compiler warning
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    TVM_TYPE_FUNCTOR_DISPATCH(TensorTypeNode);
    TVM_TYPE_FUNCTOR_DISPATCH(TypeVarNode);
    TVM_TYPE_FUNCTOR_DISPATCH(TypeConstraintNode);
    TVM_TYPE_FUNCTOR_DISPATCH(FuncTypeNode);
    TVM_TYPE_FUNCTOR_DISPATCH(TypeRelationNode);
    TVM_TYPE_FUNCTOR_DISPATCH(TupleTypeNode);
    TVM_TYPE_FUNCTOR_DISPATCH(IncompleteTypeNode);
    TVM_TYPE_FUNCTOR_DISPATCH(RelayRefTypeNode);
    TVM_TYPE_FUNCTOR_DISPATCH(GlobalTypeVarNode);
    TVM_TYPE_FUNCTOR_DISPATCH(TypeCallNode);
    TVM_TYPE_FUNCTOR_DISPATCH(TypeDataNode);
    TVM_TYPE_FUNCTOR_DISPATCH(PrimTypeNode);
    TVM_TYPE_FUNCTOR_DISPATCH(PointerTypeNode);
    return vtable;
  }
};

#undef TVM_TYPE_FUNCTOR_DISPATCH

/*!
 * \brief A type visitor that recursively visit types.
 */
class TVM_DLL TypeVisitor : public TypeFunctor<void(const Type& n)> {
 public:
  void VisitType_(const TypeVarNode* op) override;
  void VisitType_(const IncompleteTypeNode* op) override;
  void VisitType_(const TensorTypeNode* op) override;
  void VisitType_(const FuncTypeNode* op) override;
  void VisitType_(const TupleTypeNode* op) override;
  void VisitType_(const TypeRelationNode* op) override;
  void VisitType_(const RelayRefTypeNode* op) override;
  void VisitType_(const GlobalTypeVarNode* op) override;
  void VisitType_(const TypeCallNode* op) override;
  void VisitType_(const TypeDataNode* op) override;
  void VisitType_(const PrimTypeNode* op) override;
  void VisitType_(const PointerTypeNode* op) override;
};

/*!
 * \brief TypeMutator that mutates expressions.
 */
class TVM_DLL TypeMutator : public TypeFunctor<Type(const Type& n)> {
 public:
  Type VisitType(const Type& t) override;
  Type VisitType_(const TypeVarNode* op) override;
  Type VisitType_(const TensorTypeNode* op) override;
  Type VisitType_(const IncompleteTypeNode* op) override;
  Type VisitType_(const FuncTypeNode* op) override;
  Type VisitType_(const TupleTypeNode* op) override;
  Type VisitType_(const TypeRelationNode* type_rel) override;
  Type VisitType_(const RelayRefTypeNode* op) override;
  Type VisitType_(const GlobalTypeVarNode* op) override;
  Type VisitType_(const TypeCallNode* op) override;
  Type VisitType_(const TypeDataNode* op) override;
  Type VisitType_(const PrimTypeNode* op) override;
  Type VisitType_(const PointerTypeNode* op) override;

 private:
  Array<Type> MutateArray(Array<Type> arr);
};

/*!
 * \brief Bind free type variables in the type.
 * \param type The type to be updated.
 * \param args_map The binding map.
 */
Type Bind(const Type& type, const Map<TypeVar, Type>& args_map);

}  // namespace tvm
#endif  // TVM_IR_TYPE_FUNCTOR_H_
