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
 * \file tvm/relax/struct_info_functor.h
 * \brief Functors and visitors for struct info.
 */
#ifndef TVM_RELAX_STRUCT_INFO_FUNCTOR_H_
#define TVM_RELAX_STRUCT_INFO_FUNCTOR_H_

#include <tvm/node/functor.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/struct_info.h>

#include <utility>

namespace tvm {
namespace relax {

template <typename FStructInfo>
class StructInfoFunctor;

// functions to be overriden.
#define STRUCT_INFO_FUNCTOR_DEFAULT \
  { return VisitStructInfoDefault_(op, std::forward<Args>(args)...); }

#define TVM_STRUCT_INFO_FUNCTOR_DISPATCH(OP)                                                     \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {           \
    return self->VisitStructInfo_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class StructInfoFunctor<R(const StructInfo& n, Args...)> {
 private:
  using TSelf = StructInfoFunctor<R(const StructInfo& n, Args...)>;
  using FStructInfo = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~StructInfoFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const StructInfo& n, Args... args) {
    return VisitStructInfo(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitStructInfo(const StructInfo& n, Args... args) {
    ICHECK(n.defined());
    static FStructInfo vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitStructInfo_(const ObjectStructInfoNode* op,
                             Args... args) STRUCT_INFO_FUNCTOR_DEFAULT;
  virtual R VisitStructInfo_(const PrimStructInfoNode* op,
                             Args... args) STRUCT_INFO_FUNCTOR_DEFAULT;
  virtual R VisitStructInfo_(const ShapeStructInfoNode* op,
                             Args... args) STRUCT_INFO_FUNCTOR_DEFAULT;
  virtual R VisitStructInfo_(const TensorStructInfoNode* op,
                             Args... args) STRUCT_INFO_FUNCTOR_DEFAULT;
  virtual R VisitStructInfo_(const distributed::DTensorStructInfoNode* op,
                             Args... args) STRUCT_INFO_FUNCTOR_DEFAULT;
  virtual R VisitStructInfo_(const TupleStructInfoNode* op,
                             Args... args) STRUCT_INFO_FUNCTOR_DEFAULT;
  virtual R VisitStructInfo_(const FuncStructInfoNode* op,
                             Args... args) STRUCT_INFO_FUNCTOR_DEFAULT;
  virtual R VisitStructInfoDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;  // unreachable, written to stop compiler warning
  }

 private:
  // initialize the vtable.
  static FStructInfo InitVTable() {
    FStructInfo vtable;
    // Set dispatch
    TVM_STRUCT_INFO_FUNCTOR_DISPATCH(ObjectStructInfoNode);
    TVM_STRUCT_INFO_FUNCTOR_DISPATCH(PrimStructInfoNode);
    TVM_STRUCT_INFO_FUNCTOR_DISPATCH(ShapeStructInfoNode);
    TVM_STRUCT_INFO_FUNCTOR_DISPATCH(TensorStructInfoNode);
    TVM_STRUCT_INFO_FUNCTOR_DISPATCH(distributed::DTensorStructInfoNode);
    TVM_STRUCT_INFO_FUNCTOR_DISPATCH(TupleStructInfoNode);
    TVM_STRUCT_INFO_FUNCTOR_DISPATCH(FuncStructInfoNode);
    return vtable;
  }
};

#undef TVM_STRUCT_INFO_FUNCTOR_DISPATCH

/*!
 * \brief A struct info visitor.
 */
class TVM_DLL StructInfoVisitor : public StructInfoFunctor<void(const StructInfo& n)> {
 public:
  void VisitStructInfo_(const ObjectStructInfoNode* op) override;
  void VisitStructInfo_(const PrimStructInfoNode* op) override;
  void VisitStructInfo_(const ShapeStructInfoNode* op) override;
  void VisitStructInfo_(const TensorStructInfoNode* op) override;
  void VisitStructInfo_(const distributed::DTensorStructInfoNode* op) override;
  void VisitStructInfo_(const TupleStructInfoNode* op) override;
  void VisitStructInfo_(const FuncStructInfoNode* op) override;

 protected:
  // two functions to override when visit expr fields in struct info.
  virtual void VisitStructInfoExprField(const Expr& expr) {}
  virtual void VisitStructInfoExprField(const PrimExpr& expr) {}
};

/*!
 * \brief StructInfoMutator that mutates struct info.
 */
class TVM_DLL StructInfoMutator : public StructInfoFunctor<StructInfo(const StructInfo& n)> {
 public:
  StructInfo VisitStructInfo_(const ObjectStructInfoNode* op) override;
  StructInfo VisitStructInfo_(const PrimStructInfoNode* op) override;
  StructInfo VisitStructInfo_(const ShapeStructInfoNode* op) override;
  StructInfo VisitStructInfo_(const TensorStructInfoNode* op) override;
  StructInfo VisitStructInfo_(const distributed::DTensorStructInfoNode* op) override;
  StructInfo VisitStructInfo_(const TupleStructInfoNode* op) override;
  StructInfo VisitStructInfo_(const FuncStructInfoNode* op) override;

 protected:
  // two functions to override when visit expr fields in struct info.
  virtual Expr VisitStructInfoExprField(const Expr& expr) { return expr; }
  virtual PrimExpr VisitStructInfoExprField(const PrimExpr& expr) { return expr; }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_STRUCT_INFO_FUNCTOR_H_
