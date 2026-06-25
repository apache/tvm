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
 * \file tvm/relax/type_functor.h
 * \brief Functors and visitors for Relax type nodes.
 */
#ifndef TVM_RELAX_TYPE_FUNCTOR_H_
#define TVM_RELAX_TYPE_FUNCTOR_H_

#include <tvm/ir/node_functor.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/type.h>

#include <utility>

namespace tvm {
namespace relax {

template <typename FType>
class TypeFunctor;

// functions to be overriden.
#define RELAX_TYPE_FUNCTOR_DEFAULT                             \
  {                                                            \
    return VisitTypeDefault_(op, std::forward<Args>(args)...); \
  }

#define TVM_RELAX_TYPE_FUNCTOR_DISPATCH(OP)                                                 \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitType_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...);  \
  });

template <typename R, typename... Args>
class TypeFunctor<R(const Type& n, Args...)> {
 private:
  using TSelf = TypeFunctor<R(const Type& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ffi::ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~TypeFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The type node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Type& n, Args... args) { return VisitType(n, std::forward<Args>(args)...); }
  /*!
   * \brief The functor call.
   * \param n The type node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitType(const Type& n, Args... args) {
    TVM_FFI_ICHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitType_(const AnyTypeNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PrimTypeNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ShapeTypeNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TensorTypeNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const distributed::DTensorTypeNode* op,
                       Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TupleTypeNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FuncTypeNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitTypeDefault_(const ffi::Object* op, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    throw;  // unreachable, written to stop compiler warning
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(AnyTypeNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(PrimTypeNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(ShapeTypeNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(TensorTypeNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(distributed::DTensorTypeNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(TupleTypeNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(FuncTypeNode);
    vtable.Finalize();
    return vtable;
  }
};

#undef TVM_RELAX_TYPE_FUNCTOR_DISPATCH

/*!
 * \brief A type visitor.
 */
class TVM_DLL TypeVisitor : public TypeFunctor<void(const Type& n)> {
 public:
  void VisitType_(const AnyTypeNode* op) override;
  void VisitType_(const PrimTypeNode* op) override;
  void VisitType_(const ShapeTypeNode* op) override;
  void VisitType_(const TensorTypeNode* op) override;
  void VisitType_(const distributed::DTensorTypeNode* op) override;
  void VisitType_(const TupleTypeNode* op) override;
  void VisitType_(const FuncTypeNode* op) override;

 protected:
  // two functions to override when visit expr fields in type nodes.
  virtual void VisitTypeExprField(const Expr& expr) {}
  virtual void VisitTypeExprField(const PrimExpr& expr) {}
};

/*!
 * \brief TypeMutator that mutates Relax type nodes.
 */
class TVM_DLL TypeMutator : public TypeFunctor<Type(const Type& n)> {
 public:
  Type VisitType_(const AnyTypeNode* op) override;
  Type VisitType_(const PrimTypeNode* op) override;
  Type VisitType_(const ShapeTypeNode* op) override;
  Type VisitType_(const TensorTypeNode* op) override;
  Type VisitType_(const distributed::DTensorTypeNode* op) override;
  Type VisitType_(const TupleTypeNode* op) override;
  Type VisitType_(const FuncTypeNode* op) override;

 protected:
  // two functions to override when visit expr fields in type nodes.
  virtual Expr VisitTypeExprField(const Expr& expr) { return expr; }
  virtual PrimExpr VisitTypeExprField(const PrimExpr& expr) { return expr; }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TYPE_FUNCTOR_H_
