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
 * \file tvm/tirx/expr.h
 * \brief TIR expressions.
 */
// Acknowledgement: Many low-level IR nodes originate from Halide.
#ifndef TVM_TIR_EXPR_H_
#define TVM_TIR_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/node_functor.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/var.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tirx {

using IntImmNode = tvm::IntImmNode;
using FloatImmNode = tvm::FloatImmNode;

/*! \brief ffi::String constants, only used in asserts. */
class StringImmNode : public ExprNode {
 public:
  /*! \brief The constant value content. */
  ffi::String value;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StringImmNode>().def_ro("value", &StringImmNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.StringImm", StringImmNode, ExprNode);
};

/*!
 * \brief Managed reference to StringImmNode.
 * \sa StringImmNode
 */
class StringImm : public PrimExpr {
 public:
  TVM_DLL StringImm(ffi::String value, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StringImm, PrimExpr, StringImmNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmNode);
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class CastNode : public ExprNode {
 public:
  /*! \brief Original data type. */
  PrimExpr value;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CastNode>().def_ro("value", &CastNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Cast", CastNode, ExprNode);
};

/*!
 * \brief Managed reference to CastNode
 * \sa CastNode
 */
class Cast : public PrimExpr {
 public:
  TVM_DLL Cast(PrimType value_ty, PrimExpr value, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Cast, PrimExpr, CastNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CastNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class BinaryOpNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  }
  static const constexpr int _type_child_slots [[maybe_unused]] = 0;
  static const constexpr bool _type_final [[maybe_unused]] = true;
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprNode);
};

/*! \brief a + b */
class AddNode : public BinaryOpNode<AddNode> {
 public:
  static constexpr const char* _type_key = "tirx.Add";
};

/*!
 * \brief Managed reference to AddNode
 * \sa AddNode
 */
class Add : public PrimExpr {
 public:
  TVM_DLL Add(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Add, PrimExpr, AddNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AddNode);
};

/*! \brief a - b */
class SubNode : public BinaryOpNode<SubNode> {
 public:
  static constexpr const char* _type_key = "tirx.Sub";
};

/*!
 * \brief Managed reference to SubNode
 * \sa SubNode
 */
class Sub : public PrimExpr {
 public:
  TVM_DLL Sub(PrimExpr a, PrimExpr b, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Sub, PrimExpr, SubNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SubNode);
};

/*! \brief a * b */
class MulNode : public BinaryOpNode<MulNode> {
 public:
  static constexpr const char* _type_key = "tirx.Mul";
};

/*!
 * \brief Managed reference to MulNode
 * \sa MulNode
 */
class Mul : public PrimExpr {
 public:
  TVM_DLL Mul(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Mul, PrimExpr, MulNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MulNode);
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class DivNode : public BinaryOpNode<DivNode> {
 public:
  static constexpr const char* _type_key = "tirx.Div";
};

/*!
 * \brief Managed reference to DivNode
 * \sa DivNode
 */
class Div : public PrimExpr {
 public:
  TVM_DLL Div(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Div, PrimExpr, DivNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DivNode);
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class ModNode : public BinaryOpNode<ModNode> {
 public:
  static constexpr const char* _type_key = "tirx.Mod";
};

/*!
 * \brief Managed reference to ModNode
 * \sa ModNode
 */
class Mod : public PrimExpr {
 public:
  TVM_DLL Mod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Mod, PrimExpr, ModNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ModNode);
};

/*! \brief Floor division, floor(a/b) */
class FloorDivNode : public BinaryOpNode<FloorDivNode> {
 public:
  static constexpr const char* _type_key = "tirx.FloorDiv";
};

/*!
 * \brief Managed reference to FloorDivNode
 * \sa FloorDivNode
 */
class FloorDiv : public PrimExpr {
 public:
  TVM_DLL FloorDiv(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloorDiv, PrimExpr, FloorDivNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorDivNode);
};

/*! \brief The remainder of the floordiv */
class FloorModNode : public BinaryOpNode<FloorModNode> {
 public:
  static constexpr const char* _type_key = "tirx.FloorMod";
};

/*!
 * \brief Managed reference to FloorModNode
 * \sa FloorModNode
 */
class FloorMod : public PrimExpr {
 public:
  TVM_DLL FloorMod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloorMod, PrimExpr, FloorModNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorModNode);
};

/*! \brief min(a, b) */
class MinNode : public BinaryOpNode<MinNode> {
 public:
  static constexpr const char* _type_key = "tirx.Min";
};

/*!
 * \brief Managed reference to MinNode
 * \sa MinNode
 */
class Min : public PrimExpr {
 public:
  TVM_DLL Min(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Min, PrimExpr, MinNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MinNode);
};

/*! \brief max(a, b) */
class MaxNode : public BinaryOpNode<MaxNode> {
 public:
  static constexpr const char* _type_key = "tirx.Max";
};

/*!
 * \brief Managed reference to MaxNode
 * \sa MaxNode
 */
class Max : public PrimExpr {
 public:
  TVM_DLL Max(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Max, PrimExpr, MaxNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MaxNode);
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class CmpOpNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  }
  static const constexpr int _type_child_slots [[maybe_unused]] = 0;
  static const constexpr bool _type_final [[maybe_unused]] = true;
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprNode);
};

/*! \brief a == b */
class EQNode : public CmpOpNode<EQNode> {
 public:
  static constexpr const char* _type_key = "tirx.EQ";
};

/*!
 * \brief Managed reference to EQNode
 * \sa EQNode
 */
class EQ : public PrimExpr {
 public:
  TVM_DLL EQ(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(EQ, PrimExpr, EQNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EQNode);
};

/*! \brief a != b */
class NENode : public CmpOpNode<NENode> {
 public:
  static constexpr const char* _type_key = "tirx.NE";
};

/*!
 * \brief Managed reference to NENode
 * \sa NENode
 */
class NE : public PrimExpr {
 public:
  TVM_DLL NE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(NE, PrimExpr, NENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NENode);
};

/*! \brief a < b */
class LTNode : public CmpOpNode<LTNode> {
 public:
  static constexpr const char* _type_key = "tirx.LT";
};

/*!
 * \brief Managed reference to LTNode
 * \sa LTNode
 */
class LT : public PrimExpr {
 public:
  TVM_DLL LT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LT, PrimExpr, LTNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LTNode);
};

/*! \brief a <= b */
struct LENode : public CmpOpNode<LENode> {
 public:
  static constexpr const char* _type_key = "tirx.LE";
};

/*!
 * \brief Managed reference to LENode
 * \sa LENode
 */
class LE : public PrimExpr {
 public:
  TVM_DLL LE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LE, PrimExpr, LENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LENode);
};

/*! \brief a > b */
class GTNode : public CmpOpNode<GTNode> {
 public:
  static constexpr const char* _type_key = "tirx.GT";
};

/*!
 * \brief Managed reference to GTNode
 * \sa GTNode
 */
class GT : public PrimExpr {
 public:
  TVM_DLL GT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GT, PrimExpr, GTNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GTNode);
};

/*! \brief a >= b */
class GENode : public CmpOpNode<GENode> {
 public:
  static constexpr const char* _type_key = "tirx.GE";
};

/*!
 * \brief Managed reference to GENode
 * \sa GENode
 */
class GE : public PrimExpr {
 public:
  TVM_DLL GE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GE, PrimExpr, GENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GENode);
};

/*! \brief a && b */
class AndNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AndNode>().def_ro("a", &AndNode::a).def_ro("b", &AndNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.And", AndNode, ExprNode);
};

/*!
 * \brief Managed reference to AndNode
 * \sa AndNode
 */
class And : public PrimExpr {
 public:
  TVM_DLL And(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(And, PrimExpr, AndNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AndNode);
};

/*! \brief a || b */
class OrNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrNode>().def_ro("a", &OrNode::a).def_ro("b", &OrNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Or", OrNode, ExprNode);
};

/*!
 * \brief Managed reference to OrNode
 * \sa OrNode
 */
class Or : public PrimExpr {
 public:
  TVM_DLL Or(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Or, PrimExpr, OrNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OrNode);
};

/*! \brief !a */
class NotNode : public ExprNode {
 public:
  /*! \brief The input operand. */
  PrimExpr a;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NotNode>().def_ro("a", &NotNode::a);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Not", NotNode, ExprNode);
};

/*!
 * \brief Managed reference to NotNode
 * \sa NotNode
 */
class Not : public PrimExpr {
 public:
  TVM_DLL Not(PrimExpr a, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Not, PrimExpr, NotNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NotNode);
};

/*!
 * \brief return true_value if condition is true, otherwise return false_value.
 * \note Both true_value and false_value could be evaluated
 *       regardless of the condition value.
 *       Do not use it to guard against out of bound access,
 *       please use if_then_else instead.
 */
class SelectNode : public ExprNode {
 public:
  /*! \brief The condition */
  PrimExpr condition;
  /*! \brief value to be returned when condition is true. */
  PrimExpr true_value;
  /*! \brief value to be returned when condition is false. */
  PrimExpr false_value;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SelectNode>()
        .def_ro("condition", &SelectNode::condition)
        .def_ro("true_value", &SelectNode::true_value)
        .def_ro("false_value", &SelectNode::false_value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Select", SelectNode, ExprNode);
};

/*!
 * \brief Managed reference to SelectNode
 * \sa SelectNode
 */
class Select : public PrimExpr {
 public:
  TVM_DLL Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Select, PrimExpr, SelectNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SelectNode);
};

/*!
 * \brief Load value from the high dimension buffer.
 *
 * \code
 *
 *  value = buffer[i, j];
 *
 * \endcode
 * \sa BufferStore
 */
class BufferLoadNode : public ExprNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The indices location to be loaded. */
  ffi::Array<PrimExpr> indices;
  /*! \brief The predicate mask for loading values. */
  ffi::Optional<PrimExpr> predicate;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BufferLoadNode>()
        .def_ro("buffer", &BufferLoadNode::buffer)
        .def_ro("indices", &BufferLoadNode::indices)
        .def_ro("predicate", &BufferLoadNode::predicate);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.BufferLoad", BufferLoadNode, ExprNode);

 private:
  /*! \brief Set the dtype based on the buffer/indices
   *
   * Usually, the BufferLoad's dtype will be the same dtype as the
   * buffer.  This may have a different number of lanes than the
   * buffer's dtype if index values have more than 1 lane.
   *
   * This function should only be called during construction and after
   * CopyOnWrite.  Friend class used here to restrict usage.
   */
  void LegalizeDType();
  friend class BufferLoad;
  friend class CustomDatatypesLowerer;
  friend class VectorTypeRewriter;
  friend class Vectorizer;
};

/*!
 * \brief Managed reference to BufferLoadNode.
 * \sa BufferLoadNode
 */
class BufferLoad : public PrimExpr {
 public:
  TVM_DLL explicit BufferLoad(Buffer buffer, ffi::Array<PrimExpr> indices,
                              ffi::Optional<PrimExpr> predicate = std::nullopt, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferLoad, PrimExpr, BufferLoadNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferLoadNode);
};

/*!
 * \brief Load value from the result produced by the producer.
 *
 * \note This node only appears in high-level DSLs that are built on top of the TIR.
 *       It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
 *       this node before TIR transformations.
 *
 * \sa ProducerLoad, DataProducerNode
 */
class ProducerLoadNode : public ExprNode {
 public:
  /*! \brief The buffer producer. */
  DataProducer producer;
  /*! \brief The location arguments. */
  ffi::Array<PrimExpr> indices;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ProducerLoadNode>()
        .def_ro("producer", &ProducerLoadNode::producer)
        .def_ro("indices", &ProducerLoadNode::indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.ProducerLoad", ProducerLoadNode, ExprNode);
};

/*!
 * \brief Managed reference to ProducerLoadNode.
 * \sa ProducerLoadNode
 */
class ProducerLoad : public PrimExpr {
 public:
  TVM_DLL explicit ProducerLoad(DataProducer producer, ffi::Array<PrimExpr> indices,
                                Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ProducerLoad, PrimExpr, ProducerLoadNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ProducerLoadNode);
};

/*!
 * \brief Construct a vector with lanes elements
 *        where its i-th element equals base + i * stride.
 *  This is useful to construct a index for a continuous vector load.
 *
 *  Examples:
 *  - ramp(0, 1, 3) = [0, 1, 2]
 *  - ramp(1, 2, 4) = [1, 3, 5, 7]
 */
class RampNode : public ExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr base;
  /*! \brief The stride of each step. */
  PrimExpr stride;
  /*! \brief Total number of lanes. */
  PrimExpr lanes;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RampNode>()
        .def_ro("base", &RampNode::base)
        .def_ro("stride", &RampNode::stride)
        .def_ro("lanes", &RampNode::lanes);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Ramp", RampNode, ExprNode);
};

/*!
 * \brief Managed reference to RampNode
 * \sa RampNode
 */
class Ramp : public PrimExpr {
 public:
  TVM_DLL Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Ramp, PrimExpr, RampNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RampNode);
};

/*! \brief Create a vector where all the elements are value. */
class BroadcastNode : public ExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr value;
  /*! \brief The number of lanes. */
  PrimExpr lanes;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastNode>()
        .def_ro("value", &BroadcastNode::value)
        .def_ro("lanes", &BroadcastNode::lanes);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Broadcast", BroadcastNode, ExprNode);
};

/*!
 * \brief Managed reference to BroadcastNode
 * \sa BroadcastNode
 */
class Broadcast : public PrimExpr {
 public:
  TVM_DLL Broadcast(PrimExpr value, PrimExpr lanes, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Broadcast, PrimExpr, BroadcastNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BroadcastNode);
};

/*!
 * \brief Let binding. Bind var to value then evaluate body.
 */
class LetNode : public ExprNode {
 public:
  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be binded. */
  PrimExpr value;
  /*! \brief The result expression. */
  PrimExpr body;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LetNode>()
        // TODO(tqchen): use SEqHashDefNonRecursive after the next pypi tvm-ffi release
        .def_ro("var", &LetNode::var, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("value", &LetNode::value)
        .def_ro("body", &LetNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Let", LetNode, ExprNode);
};

/*!
 * \brief Managed reference to LetNode
 * \sa LetNode
 */
class Let : public PrimExpr {
 public:
  TVM_DLL Let(Var var, PrimExpr value, PrimExpr body, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Let, PrimExpr, LetNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LetNode);
};

/*!
 * \brief Shuffle instruction.
 *  vec = concat(vectors)
 *  result = (vec[indices[0]], vec[indices[1]] ...)
 */
class ShuffleNode : public ExprNode {
 public:
  /*! \brief the input vectors. */
  ffi::Array<PrimExpr> vectors;
  /*! \brief The indices of each element. */
  ffi::Array<PrimExpr> indices;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShuffleNode>()
        .def_ro("vectors", &ShuffleNode::vectors)
        .def_ro("indices", &ShuffleNode::indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Shuffle", ShuffleNode, ExprNode);
};

/*!
 * \brief Managed reference to ShuffleNode
 * \sa ShuffleNode
 */
class Shuffle : public PrimExpr {
 public:
  TVM_DLL Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span = Span());
  TVM_DLL static PrimExpr Concat(ffi::Array<PrimExpr> vectors, Span span = Span());
  TVM_DLL static PrimExpr ExtractElement(PrimExpr vector, int index, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Shuffle, PrimExpr, ShuffleNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShuffleNode);
};

// Reduce operator
/*!
 * \brief A commutative reducer node to represent a commutative
 *  binary operator with identity element
 */
class CommReducerNode : public ffi::Object {
 public:
  /*! \brief The left argument of reducer */
  ffi::Array<PrimVar> lhs;
  /*! \brief The right argument of reducer */
  ffi::Array<PrimVar> rhs;
  /*! \brief The result of reducer */
  ffi::Array<PrimExpr> result;
  /*!
   * \brief The identity element of reducer, which leaves other
   *  elements unchanged when combined with it, with respect to
   *  the binary operation of this reducer uses.
   */
  ffi::Array<PrimExpr> identity_element;
  /*! \brief Function call operator to combine a and b */
  ffi::Array<PrimExpr> operator()(ffi::Array<PrimExpr> a, ffi::Array<PrimExpr> b) const;
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CommReducerNode>()
        .def_ro("lhs", &CommReducerNode::lhs, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("rhs", &CommReducerNode::rhs, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("result", &CommReducerNode::result)
        .def_ro("identity_element", &CommReducerNode::identity_element)
        .def_ro("span", &CommReducerNode::span, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.CommReducer", CommReducerNode, ffi::Object);
};

/*!
 * \brief Managed reference to CommReducerNode
 * \sa CommReducerNode
 */
class CommReducer : public ffi::ObjectRef {
 public:
  TVM_DLL CommReducer(ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs, ffi::Array<PrimExpr> result,
                      ffi::Array<PrimExpr> identity_element, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CommReducer, ffi::ObjectRef, CommReducerNode);
};

/*! \brief Reduction operator */
class ReduceNode : public ExprNode {
 public:
  /*! \brief The commutative combiner */
  CommReducer combiner;
  /*! \brief The source operand */
  ffi::Array<PrimExpr> source;
  /*! \brief The init operand */
  ffi::Array<PrimExpr> init;
  /*! \brief The reduction axis */
  ffi::Array<IterVar> axis;
  /*!
   * \brief Predicate on the reduction
   *  Only add the body to reduction if condition is true.
   */
  PrimExpr condition;
  /*! \brief the index of this reduce node */
  int value_index;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReduceNode>()
        .def_ro("combiner", &ReduceNode::combiner)
        .def_ro("source", &ReduceNode::source)
        .def_ro("init", &ReduceNode::init)
        .def_ro("axis", &ReduceNode::axis)
        .def_ro("condition", &ReduceNode::condition)
        .def_ro("value_index", &ReduceNode::value_index);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Reduce", ReduceNode, ExprNode);
};

/*!
 * \brief Managed reference to ReduceNode
 * \sa ReduceNode
 */
class Reduce : public PrimExpr {
 public:
  TVM_DLL Reduce(CommReducer combiner, ffi::Array<PrimExpr> src, ffi::Array<IterVar> rdom,
                 PrimExpr condition, int value_index, ffi::Array<PrimExpr> init,
                 Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Reduce, PrimExpr, ReduceNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ReduceNode);
};

/*
 * \brief Template function to convert Map to unordered_map
 *  Sometimes useful for API gluing when internal uses unordered_map
 * \param dmap The container map
 * \return The corresponding unordered_map.
 * \tparam K the key of the Map.
 * \tparam V the value of the Map.
 */
template <typename K, typename V>
inline std::unordered_map<K, V> as_unordered_map(const ffi::Map<K, V>& dmap) {
  std::unordered_map<K, V> ret;
  for (auto kv : dmap) {
    ret[kv.first] = kv.second;
  }
  return ret;
}
}  // namespace tirx

namespace ffi {

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::StringImmNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::CastNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::AddNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::SubNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::MulNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::DivNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::ModNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::FloorDivNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::FloorModNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::MinNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::MaxNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::EQNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::NENode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::LTNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::LENode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::GTNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::GENode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::AndNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::OrNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::NotNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::SelectNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::BufferLoadNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::ProducerLoadNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::RampNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::BroadcastNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::LetNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::ShuffleNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, tirx::ReduceNode> = true;

template <>
inline constexpr bool use_default_type_traits_v<tvm::tirx::StringImm> = false;

template <>
struct TypeTraits<tvm::tirx::StringImm>
    : public ObjectRefWithFallbackTraitsBase<tvm::tirx::StringImm, ffi::String> {
  TVM_FFI_INLINE static tvm::tirx::StringImm ConvertFallbackValue(ffi::String value) {
    return tvm::tirx::StringImm(value);
  }
};
}  // namespace ffi
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::tirx::IterVar> : public ::tvm::ffi::ObjectPtrHash {};
}  // namespace std
#endif  // TVM_TIR_EXPR_H_
