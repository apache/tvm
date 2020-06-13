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
 * \file tvm/relay/op_strategy.h
 * \brief The Relay operator Strategy and related data structure.
 */

#ifndef TVM_RELAY_OP_STRATEGY_H_
#define TVM_RELAY_OP_STRATEGY_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/target/target.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Operator implementation that includes compute and schedule function.
 */
class OpImplementationNode : public Object {
 public:
  /*! \brief Compute function */
  FTVMCompute fcompute;
  /*! \brief Schedule function */
  FTVMSchedule fschedule;
  /*! \brief Name of the implementation */
  String name;
  /*! \brief Priority level */
  int plevel;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("plevel", &plevel);
  }

  static constexpr const char* _type_key = "relay.OpImplementation";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpImplementationNode, Object);
};

/*!
 * \brief Operator implementation class.
 */
class OpImplementation : public ObjectRef {
 public:
  /*!
   * \brief Invoke the operator compute function.
   * \param attrs The attribute of the primitive
   * \param inputs The input tensors.
   * \param out_type The output type information.
   * \return The output compute description of the operator.
   */
  TVM_DLL Array<te::Tensor> Compute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                    const Type& out_type);
  /*!
   * \brief Build the computation schedule.
   * \param attrs The attribute of the node.
   * \param outs The output tensors.
   * \param target The build target.
   * \return The computation schedule.
   */
  TVM_DLL te::Schedule Schedule(const Attrs& attrs, const Array<te::Tensor>& outs,
                                const Target& target);

  TVM_DEFINE_OBJECT_REF_METHODS(OpImplementation, ObjectRef, OpImplementationNode);
};

/*!
 * \brief Specialized implementations for operators under certain conditions.
 */
class OpSpecializationNode : public Object {
 public:
  /*! \brief List of implementations. */
  Array<OpImplementation> implementations;
  /*! \brief Condition to enable the specialization.
   *    Could be undefined to represent generic case. */
  te::SpecializedCondition condition;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("condition", &condition);
    v->Visit("implementations", &implementations);
  }

  static constexpr const char* _type_key = "relay.OpSpecialization";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpSpecializationNode, ExprNode);
};

/*!
 * \brief Operator specialization class.
 */
class OpSpecialization : public ObjectRef {
 public:
  /*!
   * \brief Add an implementation.
   * \param fcompute Compute function
   * \param fschedule Schedule function
   * \param name Name of the implementation
   * \param plevel Priority level of the implementation
   */
  TVM_DLL void AddImplementation(FTVMCompute fcompute, FTVMSchedule fschedule, String name,
                                 int plevel);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(OpSpecialization, ObjectRef, OpSpecializationNode);
};

/*!
 * \brief Operator strategy to choose implementation.
 */
class OpStrategyNode : public Object {
 public:
  /*! \brief List of operator specializations. */
  Array<OpSpecialization> specializations;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("specializations", &specializations); }

  static constexpr const char* _type_key = "relay.OpStrategy";
  TVM_DECLARE_FINAL_OBJECT_INFO(OpStrategyNode, ExprNode);
};

/*!
 * \brief Operator strategy class.
 */
class OpStrategy : public ObjectRef {
 public:
  /*!
   * \brief Add an implementation.
   * \param fcompute Compute function
   * \param fschedule Schedule function
   * \param name Name of the implementation
   * \param plevel Priority level of the implementation
   */
  TVM_DLL void AddImplementation(FTVMCompute fcompute, FTVMSchedule fschedule, String name,
                                 int plevel);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(OpStrategy, ObjectRef, OpStrategyNode);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_STRATEGY_H_
