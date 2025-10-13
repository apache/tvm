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
 * \file tvm/tir/expr.h
 * \brief TIR expressions.
 */
// Acknowledgement: Many low-level IR nodes originate from Halide.
#ifndef TVM_TIR_EXPR_H_
#define TVM_TIR_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/var.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tir {

using IntImmNode = tvm::IntImmNode;
using FloatImmNode = tvm::FloatImmNode;

/*! \brief ffi::String constants, only used in asserts. */
class StringImmNode : public PrimExprNode {
 public:
  /*! \brief The constant value content. */
  ffi::String value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StringImmNode>().def_ro("value", &StringImmNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.StringImm", StringImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference to StringImmNode.
 * \sa StringImmNode
 */
class StringImm : public PrimExpr {
 public:
  TVM_DLL StringImm(ffi::String value, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StringImm, PrimExpr, StringImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmNode);
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class CastNode : public PrimExprNode {
 public:
  /*! \brief Original data type. */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CastNode>().def_ro("value", &CastNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Cast", CastNode, PrimExprNode);
};

/*!
 * \brief Managed reference to CastNode
 * \sa CastNode
 */
class Cast : public PrimExpr {
 public:
  TVM_DLL Cast(DataType dtype, PrimExpr value, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Cast, PrimExpr, CastNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CastNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class BinaryOpNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, PrimExprNode);
};

/*! \brief a + b */
class AddNode : public BinaryOpNode<AddNode> {
 public:
  static constexpr const char* _type_key = "tir.Add";
};

/*!
 * \brief Managed reference to AddNode
 * \sa AddNode
 */
class Add : public PrimExpr {
 public:
  TVM_DLL Add(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Add, PrimExpr, AddNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AddNode);
};

/*! \brief a - b */
class SubNode : public BinaryOpNode<SubNode> {
 public:
  static constexpr const char* _type_key = "tir.Sub";
};

/*!
 * \brief Managed reference to SubNode
 * \sa SubNode
 */
class Sub : public PrimExpr {
 public:
  TVM_DLL Sub(PrimExpr a, PrimExpr b, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Sub, PrimExpr, SubNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SubNode);
};

/*! \brief a * b */
class MulNode : public BinaryOpNode<MulNode> {
 public:
  static constexpr const char* _type_key = "tir.Mul";
};

/*!
 * \brief Managed reference to MulNode
 * \sa MulNode
 */
class Mul : public PrimExpr {
 public:
  TVM_DLL Mul(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Mul, PrimExpr, MulNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MulNode);
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class DivNode : public BinaryOpNode<DivNode> {
 public:
  static constexpr const char* _type_key = "tir.Div";
};

/*!
 * \brief Managed reference to DivNode
 * \sa DivNode
 */
class Div : public PrimExpr {
 public:
  TVM_DLL Div(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Div, PrimExpr, DivNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DivNode);
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class ModNode : public BinaryOpNode<ModNode> {
 public:
  static constexpr const char* _type_key = "tir.Mod";
};

/*!
 * \brief Managed reference to ModNode
 * \sa ModNode
 */
class Mod : public PrimExpr {
 public:
  TVM_DLL Mod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Mod, PrimExpr, ModNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ModNode);
};

/*! \brief Floor division, floor(a/b) */
class FloorDivNode : public BinaryOpNode<FloorDivNode> {
 public:
  static constexpr const char* _type_key = "tir.FloorDiv";
};

/*!
 * \brief Managed reference to FloorDivNode
 * \sa FloorDivNode
 */
class FloorDiv : public PrimExpr {
 public:
  TVM_DLL FloorDiv(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloorDiv, PrimExpr, FloorDivNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorDivNode);
};

/*! \brief The remainder of the floordiv */
class FloorModNode : public BinaryOpNode<FloorModNode> {
 public:
  static constexpr const char* _type_key = "tir.FloorMod";
};

/*!
 * \brief Managed reference to FloorModNode
 * \sa FloorModNode
 */
class FloorMod : public PrimExpr {
 public:
  TVM_DLL FloorMod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloorMod, PrimExpr, FloorModNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorModNode);
};

/*! \brief min(a, b) */
class MinNode : public BinaryOpNode<MinNode> {
 public:
  static constexpr const char* _type_key = "tir.Min";
};

/*!
 * \brief Managed reference to MinNode
 * \sa MinNode
 */
class Min : public PrimExpr {
 public:
  TVM_DLL Min(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Min, PrimExpr, MinNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MinNode);
};

/*! \brief max(a, b) */
class MaxNode : public BinaryOpNode<MaxNode> {
 public:
  static constexpr const char* _type_key = "tir.Max";
};

/*!
 * \brief Managed reference to MaxNode
 * \sa MaxNode
 */
class Max : public PrimExpr {
 public:
  TVM_DLL Max(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Max, PrimExpr, MaxNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MaxNode);
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class CmpOpNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, PrimExprNode);
};

/*! \brief a == b */
class EQNode : public CmpOpNode<EQNode> {
 public:
  static constexpr const char* _type_key = "tir.EQ";
};

/*!
 * \brief Managed reference to EQNode
 * \sa EQNode
 */
class EQ : public PrimExpr {
 public:
  TVM_DLL EQ(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(EQ, PrimExpr, EQNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EQNode);
};

/*! \brief a != b */
class NENode : public CmpOpNode<NENode> {
 public:
  static constexpr const char* _type_key = "tir.NE";
};

/*!
 * \brief Managed reference to NENode
 * \sa NENode
 */
class NE : public PrimExpr {
 public:
  TVM_DLL NE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(NE, PrimExpr, NENode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NENode);
};

/*! \brief a < b */
class LTNode : public CmpOpNode<LTNode> {
 public:
  static constexpr const char* _type_key = "tir.LT";
};

/*!
 * \brief Managed reference to LTNode
 * \sa LTNode
 */
class LT : public PrimExpr {
 public:
  TVM_DLL LT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LT, PrimExpr, LTNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LTNode);
};

/*! \brief a <= b */
struct LENode : public CmpOpNode<LENode> {
 public:
  static constexpr const char* _type_key = "tir.LE";
};

/*!
 * \brief Managed reference to LENode
 * \sa LENode
 */
class LE : public PrimExpr {
 public:
  TVM_DLL LE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LE, PrimExpr, LENode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LENode);
};

/*! \brief a > b */
class GTNode : public CmpOpNode<GTNode> {
 public:
  static constexpr const char* _type_key = "tir.GT";
};

/*!
 * \brief Managed reference to GTNode
 * \sa GTNode
 */
class GT : public PrimExpr {
 public:
  TVM_DLL GT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GT, PrimExpr, GTNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GTNode);
};

/*! \brief a >= b */
class GENode : public CmpOpNode<GENode> {
 public:
  static constexpr const char* _type_key = "tir.GE";
};

/*!
 * \brief Managed reference to GENode
 * \sa GENode
 */
class GE : public PrimExpr {
 public:
  TVM_DLL GE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GE, PrimExpr, GENode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GENode);
};

/*! \brief a && b */
class AndNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AndNode>().def_ro("a", &AndNode::a).def_ro("b", &AndNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.And", AndNode, PrimExprNode);
};

/*!
 * \brief Managed reference to AndNode
 * \sa AndNode
 */
class And : public PrimExpr {
 public:
  TVM_DLL And(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(And, PrimExpr, AndNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AndNode);
};

/*! \brief a || b */
class OrNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrNode>().def_ro("a", &OrNode::a).def_ro("b", &OrNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Or", OrNode, PrimExprNode);
};

/*!
 * \brief Managed reference to OrNode
 * \sa OrNode
 */
class Or : public PrimExpr {
 public:
  TVM_DLL Or(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Or, PrimExpr, OrNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OrNode);
};

/*! \brief !a */
class NotNode : public PrimExprNode {
 public:
  /*! \brief The input operand. */
  PrimExpr a;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NotNode>().def_ro("a", &NotNode::a);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Not", NotNode, PrimExprNode);
};

/*!
 * \brief Managed reference to NotNode
 * \sa NotNode
 */
class Not : public PrimExpr {
 public:
  TVM_DLL Not(PrimExpr a, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Not, PrimExpr, NotNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NotNode);
};

/*!
 * \brief return true_value if condition is true, otherwise return false_value.
 * \note Both true_value and false_value could be evaluated
 *       regardless of the condition value.
 *       Do not use it to guard against out of bound access,
 *       please use if_then_else instead.
 */
class SelectNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Select", SelectNode, PrimExprNode);
};

/*!
 * \brief Managed reference to SelectNode
 * \sa SelectNode
 */
class Select : public PrimExpr {
 public:
  TVM_DLL Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Select, PrimExpr, SelectNode);
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
class BufferLoadNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.BufferLoad", BufferLoadNode, PrimExprNode);

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
class ProducerLoadNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.ProducerLoad", ProducerLoadNode, PrimExprNode);
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
class RampNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Ramp", RampNode, PrimExprNode);
};

/*!
 * \brief Managed reference to RampNode
 * \sa RampNode
 */
class Ramp : public PrimExpr {
 public:
  TVM_DLL Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Ramp, PrimExpr, RampNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RampNode);
};

/*! \brief Create a vector where all the elements are value. */
class BroadcastNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Broadcast", BroadcastNode, PrimExprNode);
};

/*!
 * \brief Managed reference to BroadcastNode
 * \sa BroadcastNode
 */
class Broadcast : public PrimExpr {
 public:
  TVM_DLL Broadcast(PrimExpr value, PrimExpr lanes, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Broadcast, PrimExpr, BroadcastNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BroadcastNode);
};

/*!
 * \brief Let binding. Bind var to value then evaluate body.
 */
class LetNode : public PrimExprNode {
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
        .def_ro("var", &LetNode::var, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("value", &LetNode::value)
        .def_ro("body", &LetNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Let", LetNode, PrimExprNode);
};

/*!
 * \brief Managed reference to LetNode
 * \sa LetNode
 */
class Let : public PrimExpr {
 public:
  TVM_DLL Let(Var var, PrimExpr value, PrimExpr body, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Let, PrimExpr, LetNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LetNode);
};

/*!
 * \brief Call node.
 */
class CallNode : public PrimExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be tvm::Op which corresponds to the primitive operators(intrinsics).
   *  - It can also be another function in the IRModule (GlobalVar).
   */
  RelaxExpr op;

  /*! \brief The arguments. */
  ffi::Array<PrimExpr> args;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallNode>().def_ro("op", &CallNode::op).def_ro("args", &CallNode::args);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Call", CallNode, PrimExprNode);
};

/*!
 * \brief Managed reference to CallNode
 * \sa CallNode
 */
class Call : public PrimExpr {
 public:
  TVM_DLL Call(DataType dtype, RelaxExpr op, ffi::Array<PrimExpr> args, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Call, PrimExpr, CallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
};

/*!
 * \brief Shuffle instruction.
 *  vec = concat(vectors)
 *  result = (vec[indices[0]], vec[indices[1]] ...)
 */
class ShuffleNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Shuffle", ShuffleNode, PrimExprNode);
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
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShuffleNode);
};

// Reduce operator
/*!
 * \brief A commutative reducer node to represent a commutative
 *  binary operator with identity element
 */
class CommReducerNode : public Object {
 public:
  /*! \brief The left argument of reducer */
  ffi::Array<Var> lhs;
  /*! \brief The right argument of reducer */
  ffi::Array<Var> rhs;
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
        .def_ro("lhs", &CommReducerNode::lhs, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("rhs", &CommReducerNode::rhs, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("result", &CommReducerNode::result)
        .def_ro("identity_element", &CommReducerNode::identity_element)
        .def_ro("span", &CommReducerNode::span, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.CommReducer", CommReducerNode, Object);
};

/*!
 * \brief Managed reference to CommReducerNode
 * \sa CommReducerNode
 */
class CommReducer : public ObjectRef {
 public:
  TVM_DLL CommReducer(ffi::Array<Var> lhs, ffi::Array<Var> rhs, ffi::Array<PrimExpr> result,
                      ffi::Array<PrimExpr> identity_element, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CommReducer, ObjectRef, CommReducerNode);
};

/*! \brief Reduction operator */
class ReduceNode : public PrimExprNode {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tir.Reduce", ReduceNode, PrimExprNode);
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
}  // namespace tir

namespace ffi {

template <>
inline constexpr bool use_default_type_traits_v<tvm::tir::StringImm> = false;

template <>
struct TypeTraits<tvm::tir::StringImm>
    : public ObjectRefWithFallbackTraitsBase<tvm::tir::StringImm, ffi::String> {
  TVM_FFI_INLINE static tvm::tir::StringImm ConvertFallbackValue(ffi::String value) {
    return tvm::tir::StringImm(value);
  }
};
}  // namespace ffi
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::tir::IterVar> : public ::tvm::ObjectPtrHash {};
}  // namespace std
#endif  // TVM_TIR_EXPR_H_
