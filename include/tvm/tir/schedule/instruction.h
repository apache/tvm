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
#ifndef TVM_TIR_SCHEDULE_INSTRUCTION_H_
#define TVM_TIR_SCHEDULE_INSTRUCTION_H_

#include <tvm/node/reflection.h>

#include <utility>

namespace tvm {

// Forward declaration
template <typename, typename>
class AttrRegistry;

namespace tir {

// Forward declaration
class Schedule;

/*!
 * \brief Type of the functor that applies the instruction to a TensorIR schedule
 * \param sch The schedule to be applied on
 * \param inputs The input random variables
 * \param attrs Instruction attributes
 * \param decision Decisions made on the instruction
 * \return The functor returns an array of output random variables
 */
using FInstructionApply = runtime::TypedPackedFunc<Array<ObjectRef>(
    Schedule sch, const Array<ObjectRef>& inputs, const Array<ObjectRef>& attrs,
    const Optional<ObjectRef>& decision)>;

/*!
 * \brief Type of the functor that converts the instruction to a statement in python syntax
 * \param inputs Names of the input random variables
 * \param attrs Instruction attributes
 * \param decisions Decisions made on the instruction
 * \param outputs Names of the output random variables
 * \return A string representing the python api call
 */
using FInstructionAsPython = runtime::TypedPackedFunc<String(
    const Array<ObjectRef>& inputs, const Array<ObjectRef>& attrs,
    const Optional<ObjectRef>& decision, const Array<String>& outputs)>;

/*!
 * \brief Type of the functor that serialize its attributes to JSON
 * \param attrs The attributes to be serialized
 * \return An array, serialized attributes
 * \note This functor is nullable
 */
using FInstructionAttrsAsJSON = runtime::TypedPackedFunc<ObjectRef(Array<ObjectRef> attrs)>;

/*!
 * \brief Type of the functor that deserialize its attributes from JSON
 * \param json_attrs The attributes to be serialized
 * \return An array, deserialized attributes
 * \note This functor is nullable
 */
using FInstructionAttrsFromJSON = runtime::TypedPackedFunc<Array<ObjectRef>(ObjectRef json_attrs)>;

/*!
 * \brief Kind of an instruction, e.g. Split, Reorder, etc.
 * Besides the name, every kind of instruction has its own properties, including:
 * 1) A boolean indicating if the instruction is pure, i.e. change nothing in the schedule state
 * 2) A functor that applies the instruction to a TensorIR schedule
 * 3) A functor that converts the instruction to a statement in python syntax
 * 4) A functor that serialize its attributes to JSON
 * 5) A functor that deserialize its attributes from JSON
 *
 * Unlike `tvm::OpNode`, `InstructionKindNode` doesn't support unstructured properties,
 * mainly because there is no such usecase yet to add any other property.
 */
class InstructionKindNode : public runtime::Object {
 public:
  /*! \brief The name of a kind of instructions */
  String name;
  /*!
   * \brief Indicates if the instruction is pure, i.e. removing it alone doesn't mutate the schedule
   * state. For example, the instruction `GetBlock` is pure because it changes
   * nothing, while `ComputeInline` is not because removing it leads to a different resulting
   * schedule.
   */
  bool is_pure{false};
  /*! \brief A functor that applies the instruction to a TensorIR schedule */
  FInstructionApply f_apply_to_schedule{nullptr};
  /*! \brief A functor that converts the instruction to a statement in python syntax */
  FInstructionAsPython f_as_python{nullptr};
  /*!
   * \brief A functor that serialize its attributes to JSON
   * \note If the functor is null, it means no conversion is needed
   */
  FInstructionAttrsAsJSON f_attrs_as_json{nullptr};
  /*!
   * \brief A functor that deserialize its attributes from JSON
   * \note If the functor is null, it means no conversion is needed
   */
  FInstructionAttrsFromJSON f_attrs_from_json{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("_is_pure", &is_pure);
    // not visited: f_apply_to_schedule
    // not visited: f_as_python
    // not visited: f_attrs_as_json
    // not visited: f_attrs_from_json
  }

  /*! \brief Checks if the instruction kind is EnterPostproc */
  bool IsPostproc() const;

  static constexpr const char* _type_key = "tir.InstructionKind";
  TVM_DECLARE_FINAL_OBJECT_INFO(InstructionKindNode, runtime::Object);
};

/*!
 * \brief Managed reference to InstructionKindNode
 * \sa InstructionKindNode
 */
class InstructionKind : public runtime::ObjectRef {
 public:
  /*!
   * \brief Retrieve an InstructionKind using its name
   * \param name The registered name of the InstructionKind
   * \return The InstructionKind retrieved
   */
  static InstructionKind Get(const String& name);
  TVM_DEFINE_OBJECT_REF_METHODS(InstructionKind, runtime::ObjectRef, InstructionKindNode);
};

/*! \brief Schedule instructions each corresponds to a schedule primitive */
class InstructionNode : public runtime::Object {
 public:
  /*! \brief The kind of the instruction */
  InstructionKind kind;
  /*!
   * \brief The input random variables of the instruction, and the type of each element can be one
   * of the following:
   * - BlockRV
   * - LoopRV
   * - ExprRV
   * - FloatImm
   * - IntImm
   * - String
   * - null pointer
   */
  Array<ObjectRef> inputs;
  /*!
   * \brief The attributes of the instruction. Similar to attributes of an operator,
   * attributes of an instruction are arbitrary constant metadata required by the instructions.
   * For example, the name of the block to be retrieved in `GetBlock`.
   */
  Array<ObjectRef> attrs;
  /*! \brief The output random variables of the instruction, and the type of each element can be one
   * of the following:
   * - BlockRV
   * - LoopRV
   * - ExprRV, atomic variables only, won't be constants or composite PrimExpr
   */
  Array<ObjectRef> outputs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("kind", &kind);
    v->Visit("inputs", &inputs);
    v->Visit("attrs", &attrs);
    v->Visit("outputs", &outputs);
  }

  static constexpr const char* _type_key = "tir.Instruction";
  TVM_DECLARE_FINAL_OBJECT_INFO(InstructionNode, runtime::Object);
};

/*!
 * \brief Managed reference to InstructionNode
 * \sa InstructionNode
 */
class Instruction : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param kind The kind of the instruction
   * \param inputs The input random variables of the instruction
   * \param attrs The attributes of the instruction
   * \param outputs The output random variables of the instruction
   */
  explicit Instruction(InstructionKind kind, Array<ObjectRef> inputs, Array<ObjectRef> attrs,
                       Array<ObjectRef> outputs);

  TVM_DEFINE_OBJECT_REF_METHODS(Instruction, runtime::ObjectRef, InstructionNode);
};

/*!
 * \brief A helper macro to register InstructionKind, only used in `TVM_REGISTER_INST_KIND`
 * \note This macro is not user-facing.
 * \sa TVM_REGISTER_INST_KIND
 */
#define TVM_INST_KIND_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::tir::InstructionKindRegEntry& __make_##InstructionKind

/*!
 * \brief Register an InstructionKind
 * \param InstructionKindName The name of the InstructionKind
 *
 * Example:
 *
 * \code
 *
 * TVM_REGISTER_INST_KIND("ComputeInline")
 *     .set_is_pure(false)
 *     .set_apply_to_schedule(ApplyToSchedule)
 *     .set_attrs_as_json(AttrsAsJSON)
 *     .set_attrs_from_json(AttrsFromJSON)
 *     .set_as_python(AsPython);
 *
 * \endcode
 */
#define TVM_REGISTER_INST_KIND(InstructionKindName)             \
  TVM_STR_CONCAT(TVM_INST_KIND_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::tir::InstructionKindRegEntry::RegisterOrGet(InstructionKindName).set_name()

/*! \brief An entry in the registry of InstructionKind */
class InstructionKindRegEntry {
 public:
  static InstructionKindRegEntry& RegisterOrGet(const String& name);

  InstructionKindRegEntry& set_name() {
    get_mutable()->name = this->name;
    return *this;
  }

  InstructionKindRegEntry& set_is_pure(bool is_pure) {
    get_mutable()->is_pure = is_pure;
    return *this;
  }

  InstructionKindRegEntry& set_apply_to_schedule(FInstructionApply f_apply_to_schedule) {
    get_mutable()->f_apply_to_schedule = std::move(f_apply_to_schedule);
    return *this;
  }

  InstructionKindRegEntry& set_as_python(FInstructionAsPython f_as_python) {
    get_mutable()->f_as_python = std::move(f_as_python);
    return *this;
  }

  InstructionKindRegEntry& set_attrs_as_json(FInstructionAttrsAsJSON f_attrs_as_json) {
    get_mutable()->f_attrs_as_json = std::move(f_attrs_as_json);
    return *this;
  }

  InstructionKindRegEntry& set_attrs_from_json(FInstructionAttrsFromJSON f_attrs_from_json) {
    get_mutable()->f_attrs_from_json = std::move(f_attrs_from_json);
    return *this;
  }

 private:
  /*! \brief Private constructor, used only by AttrRegistry */
  explicit InstructionKindRegEntry(uint32_t reg_index);
  /*! \brief Get the mutable reference to the internal InstructionKind */
  InstructionKindNode* get_mutable() const {
    return const_cast<InstructionKindNode*>(inst_kind_.get());
  }

  /*! \brief The name of the registry entry */
  String name;
  /*! \brief The instruction kind */
  InstructionKind inst_kind_;
  template <typename, typename>
  friend class ::tvm::AttrRegistry;
  friend class InstructionKind;
};

}  // namespace tir
}  // namespace tvm

#endif  //  TVM_TIR_SCHEDULE_INSTRUCTION_H_
