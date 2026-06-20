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
class TypeFunctor<R(const StructInfo& n, Args...)> {
 private:
  using TSelf = TypeFunctor<R(const StructInfo& n, Args...)>;
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
  R operator()(const StructInfo& n, Args... args) {
    return VisitType(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The type node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitType(const StructInfo& n, Args... args) {
    TVM_FFI_ICHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitType_(const ObjectStructInfoNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const PrimStructInfoNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const ShapeStructInfoNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TensorStructInfoNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const distributed::DTensorTypeNode* op,
                       Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const TupleStructInfoNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitType_(const FuncStructInfoNode* op, Args... args) RELAX_TYPE_FUNCTOR_DEFAULT;
  virtual R VisitTypeDefault_(const ffi::Object* op, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    throw;  // unreachable, written to stop compiler warning
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(ObjectStructInfoNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(PrimStructInfoNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(ShapeStructInfoNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(TensorStructInfoNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(distributed::DTensorTypeNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(TupleStructInfoNode);
    TVM_RELAX_TYPE_FUNCTOR_DISPATCH(FuncStructInfoNode);
    vtable.Finalize();
    return vtable;
  }
};

#undef TVM_RELAX_TYPE_FUNCTOR_DISPATCH

/*!
 * \brief A type visitor.
 */
class TVM_DLL TypeVisitor : public TypeFunctor<void(const StructInfo& n)> {
 public:
  void VisitType_(const ObjectStructInfoNode* op) override;
  void VisitType_(const PrimStructInfoNode* op) override;
  void VisitType_(const ShapeStructInfoNode* op) override;
  void VisitType_(const TensorStructInfoNode* op) override;
  void VisitType_(const distributed::DTensorTypeNode* op) override;
  void VisitType_(const TupleStructInfoNode* op) override;
  void VisitType_(const FuncStructInfoNode* op) override;

 protected:
  // two functions to override when visit expr fields in type nodes.
  virtual void VisitStructInfoExprField(const Expr& expr) {}
  virtual void VisitStructInfoExprField(const PrimExpr& expr) {}
};

/*!
 * \brief TypeMutator that mutates Relax type nodes.
 */
class TVM_DLL TypeMutator : public TypeFunctor<StructInfo(const StructInfo& n)> {
 public:
  StructInfo VisitType_(const ObjectStructInfoNode* op) override;
  StructInfo VisitType_(const PrimStructInfoNode* op) override;
  StructInfo VisitType_(const ShapeStructInfoNode* op) override;
  StructInfo VisitType_(const TensorStructInfoNode* op) override;
  StructInfo VisitType_(const distributed::DTensorTypeNode* op) override;
  StructInfo VisitType_(const TupleStructInfoNode* op) override;
  StructInfo VisitType_(const FuncStructInfoNode* op) override;

 protected:
  // two functions to override when visit expr fields in type nodes.
  virtual Expr VisitStructInfoExprField(const Expr& expr) { return expr; }
  virtual PrimExpr VisitStructInfoExprField(const PrimExpr& expr) { return expr; }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TYPE_FUNCTOR_H_
