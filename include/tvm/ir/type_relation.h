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
 * \file tvm/ir/type_relation.h
 * \brief Type relation function for type checking.
 */
#ifndef TVM_IR_TYPE_RELATION_H_
#define TVM_IR_TYPE_RELATION_H_

#include <tvm/ir/type.h>
#include <tvm/attrs.h>

namespace tvm {

// TODO(tqchen): remove after migrate Module to ir.
class IRModule;


/*!
 * \brief reporter that reports back to the
 *  type resolution information.
 */
class TypeReporterNode : public Object {
 public:
  /*!
   * \brief Create a type equality constraint.
   *
   *  The "assign direction" acts as a hint to the solver
   *  showing that it is more likely to resolve dst by src.
   *  But it is possible for the solver to resolve src by dst as well.
   */
  TVM_DLL virtual void Assign(const Type& dst, const Type& src) = 0;

  /*!
   * \brief assert shape expression comparison.
   * \note Use assert only if any of the condition input is symbolic.
   * \param cond The condition of operation.
   * \return false if assertation can be proven to have failed
   *      true if solver can still proceed.
   */
  TVM_DLL virtual bool Assert(const PrimExpr& cond)= 0;
  /*!
   * \brief assert shape expression equals each other.
   * \param lhs The left operand.
   * \param rhs The right operand.
   * \return false if assertation can be proven to have failed
   *      true if solver can still proceed.
   */
  TVM_DLL virtual bool AssertEQ(const PrimExpr& lhs, const PrimExpr& rhs) = 0;

  /*!
   * \brief Set the location at which to report unification errors.
   * \param ref The program node to report the error.
   */
  TVM_DLL virtual void SetLocation(const ObjectRef& ref) = 0;

  /*!
   * \brief Retrieve the current global module.
   * \return The global module.
   */
  TVM_DLL virtual IRModule GetModule() = 0;

  // solver is not serializable.
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relay.TypeReporter";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeReporterNode, Object);
};

/*!
 * \brief Container class of TypeReporter.
 * \sa TypeReporterNode
 */
class TypeReporter : public ObjectRef {
 public:
  TypeReporter() {}
  explicit TypeReporter(::tvm::ObjectPtr<::tvm::Object> n) : ObjectRef(n) {
  }
  TypeReporterNode* operator->() const {
    return const_cast<TypeReporterNode*>(
        static_cast<const TypeReporterNode*>(get()));
  }
  using ContainerType = TypeReporterNode;
};

/*!
 * \brief User defined type constraint function.
 *
 * If the input type information can be used to fully decide
 * the IncompleteTypes, then the function should call
 * reporter.Assign to report the new types, and return true.
 * Otherwise, the function should return false.
 *
 * \param args The arguments to the relation.
 *   The types are stored in the form of
 *   [input_type_0, input_type_1, ... input_type_n,
 *    output_type_0, output_type_1, ... output_type_m]
 *
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return false if This relation cannot be resolved.
 *   true if this relation has been resolved.
 */
using TypeRelationFn =
    TypedEnvFunc<bool(const Array<Type>& args,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter)>;

/*!
 * \brief User defined type relation, is an input-output relation on types.
 */
class TypeRelation;
/*!
 * \brief TypeRelation container.
 * \note This node is not directly serializable.
 * The type function need to be lookedup in the module.
 */
class TypeRelationNode : public TypeConstraintNode {
 public:
  /*!
   * \brief The function on input and output variables which
   *  this is not directly serializable,
   *  need to be looked-up in the module.
   */
  TypeRelationFn func;
  /*! \brief The type arguments to the type function. */
  tvm::Array<Type> args;
  /*! \brief Number of inputs arguments */
  int num_inputs;
  /*! \brief Attributes to the relation function */
  Attrs attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("args", &args);
    v->Visit("num_inputs", &num_inputs);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeRelation make(TypeRelationFn func,
                                   Array<Type> args,
                                   int num_args,
                                   Attrs attrs);

  static constexpr const char* _type_key = "relay.TypeRelation";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeRelationNode, TypeConstraintNode);
};

class TypeRelation : public TypeConstraint {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TypeRelation, TypeConstraint, TypeRelationNode);
};
}  // namespace tvm
#endif  // TVM_IR_TYPE_RELATION_H_
