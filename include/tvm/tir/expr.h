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

#include <tvm/node/node.h>
#include <tvm/node/container.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/data_type.h>
#include <tvm/ir/expr.h>
#include <tvm/tir/var.h>
#include <tvm/tir/buffer.h>

#include <string>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <limits>
#include <utility>

namespace tvm {
namespace tir {

using IntImmNode = tvm::IntImmNode;
using FloatImmNode = tvm::FloatImmNode;

/*! \brief String constants, only used in asserts. */
class StringImmNode : public PrimExprNode {
 public:
  /*! \brief The constant value content. */
  std::string value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const StringImmNode* other, SEqualReducer equal) const {
    return equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(value);
  }

  TVM_DLL PrimExpr static make(std::string value);

  static constexpr const char* _type_key = "StringImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringImmNode, PrimExprNode);
};

class StringImm : public PrimExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(StringImm, PrimExpr, StringImmNode);
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class CastNode : public PrimExprNode {
 public:
  /*! \brief Original data type. */
  PrimExpr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const CastNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  TVM_DLL static PrimExpr make(DataType t, PrimExpr v);

  static constexpr const char* _type_key = "Cast";
  TVM_DECLARE_FINAL_OBJECT_INFO(CastNode, PrimExprNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template<typename T>
class BinaryOpNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(a, other->a) &&
        equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  static PrimExpr make(PrimExpr a, PrimExpr b) {
    CHECK(a.defined()) << "ValueError: a is undefined\n";
    CHECK(b.defined()) << "ValueError: b is undefined\n";
    CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n";
    ObjectPtr<T> node = make_object<T>();
    node->dtype = a.dtype();
    node->a = std::move(a);
    node->b = std::move(b);
    return PrimExpr(node);
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(T, PrimExprNode);
};

/*! \brief a + b */
class AddNode : public BinaryOpNode<AddNode> {
 public:
  static constexpr const char* _type_key = "Add";
};

/*! \brief a - b */
class SubNode : public BinaryOpNode<SubNode> {
 public:
  static constexpr const char* _type_key = "Sub";
};

/*! \brief a * b */
class MulNode : public BinaryOpNode<MulNode> {
 public:
  static constexpr const char* _type_key = "Mul";
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class DivNode : public BinaryOpNode<DivNode> {
 public:
  static constexpr const char* _type_key = "Div";
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class ModNode : public BinaryOpNode<ModNode> {
 public:
  static constexpr const char* _type_key = "Mod";
};

/*! \brief Floor division, floor(a/b) */
class FloorDivNode : public BinaryOpNode<FloorDivNode> {
 public:
  static constexpr const char* _type_key = "FloorDiv";
};

/*! \brief The remainder of the floordiv */
class FloorModNode : public BinaryOpNode<FloorModNode> {
 public:
  static constexpr const char* _type_key = "FloorMod";
};

/*! \brief min(a, b) */
class MinNode : public BinaryOpNode<MinNode> {
 public:
  static constexpr const char* _type_key = "Min";
};

/*! \brief max(a, b) */
class MaxNode : public BinaryOpNode<MaxNode> {
 public:
  static constexpr const char* _type_key = "Max";
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template<typename T>
class CmpOpNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(a, other->a) &&
        equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  static PrimExpr make(PrimExpr a, PrimExpr b) {
    CHECK(a.defined()) << "ValueError: a is undefined\n";
    CHECK(b.defined()) << "ValueError: b is undefined\n";
    CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n";
    ObjectPtr<T> node = make_object<T>();
    node->dtype = DataType::Bool(a.dtype().lanes());
    node->a = std::move(a);
    node->b = std::move(b);
    return PrimExpr(node);
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(T, PrimExprNode);
};

/*! \brief a == b */
class EQNode : public CmpOpNode<EQNode> {
 public:
  static constexpr const char* _type_key = "EQ";
};

/*! \brief a != b */
class NENode : public CmpOpNode<NENode> {
 public:
  static constexpr const char* _type_key = "NE";
};

/*! \brief a < b */
class LTNode : public CmpOpNode<LTNode> {
 public:
  static constexpr const char* _type_key = "LT";
};

/*! \brief a <= b */
struct LENode : public CmpOpNode<LENode> {
 public:
  static constexpr const char* _type_key = "LE";
};

/*! \brief a > b */
class GTNode : public CmpOpNode<GTNode> {
 public:
  static constexpr const char* _type_key = "GT";
};

/*! \brief a >= b */
class GENode : public CmpOpNode<GENode> {
 public:
  static constexpr const char* _type_key = "GE";
};

/*! \brief a && b */
class AndNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const AndNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(a, other->a) &&
        equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  TVM_DLL static PrimExpr make(PrimExpr a, PrimExpr b);

  static constexpr const char* _type_key = "And";
  TVM_DECLARE_FINAL_OBJECT_INFO(AndNode, PrimExprNode);
};

/*! \brief a || b */
class OrNode : public PrimExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const OrNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(a, other->a) &&
        equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  TVM_DLL static PrimExpr make(PrimExpr a, PrimExpr b);

  static constexpr const char* _type_key = "Or";
  TVM_DECLARE_FINAL_OBJECT_INFO(OrNode, PrimExprNode);
};

/*! \brief !a */
class NotNode : public PrimExprNode {
 public:
  /*! \brief The input operand. */
  PrimExpr a;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("a", &a);
  }

  bool SEqualReduce(const NotNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(a, other->a);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
  }

  TVM_DLL static PrimExpr make(PrimExpr a);

  static constexpr const char* _type_key = "Not";
  TVM_DECLARE_FINAL_OBJECT_INFO(NotNode, PrimExprNode);
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("condition", &condition);
    v->Visit("true_value", &true_value);
    v->Visit("false_value", &false_value);
  }

  bool SEqualReduce(const SelectNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(condition, other->condition) &&
        equal(true_value, other->true_value) &&
        equal(false_value, other->false_value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(condition);
    hash_reduce(true_value);
    hash_reduce(false_value);
  }

  TVM_DLL static PrimExpr make(PrimExpr condition, PrimExpr true_value, PrimExpr false_value);

  static constexpr const char* _type_key = "Select";
  TVM_DECLARE_FINAL_OBJECT_INFO(SelectNode, PrimExprNode);
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
  Array<PrimExpr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("buffer", &buffer);
    v->Visit("indices", &indices);
  }

  bool SEqualReduce(const BufferLoadNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(buffer, other->buffer) &&
        equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(buffer);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "BufferLoad";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferLoadNode, PrimExprNode);
};

class BufferLoad : public PrimExpr {
 public:
  TVM_DLL explicit BufferLoad(Buffer buffer,
                              Array<PrimExpr> indices);
  TVM_DEFINE_OBJECT_REF_METHODS(BufferLoad, PrimExpr, BufferLoadNode);
};

/*!
 * \brief Load the value from buffer_var.
 *
 *  Equivalent to ((DType*)buffer_var)[index]
 *  where DType is the type specified by type().element_of().
 *
 *  For example, if type = float32x3, then the load will corresponds to
 *
 * \code
 *
 *  auto buffer = static_cast<float*>(buffer_var);
 *  auto loaded_val = float32x3(buffer[index.v0], buffer[index.v1], buffer[index.v2]);
 *
 * \endcode
 */
class LoadNode : public PrimExprNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The index locations to be loaded. */
  PrimExpr index;
  /*! \brief The predicate to mask which lanes would be loaded. */
  PrimExpr predicate;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("buffer_var", &buffer_var);
    v->Visit("index", &index);
    v->Visit("predicate", &predicate);
  }

  bool SEqualReduce(const LoadNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(buffer_var, other->buffer_var) &&
        equal(index, other->index) &&
        equal(predicate, other->predicate);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(buffer_var);
    hash_reduce(index);
    hash_reduce(predicate);
  }

  TVM_DLL static PrimExpr make(DataType dtype, Var buffer_var, PrimExpr index, PrimExpr predicate);

  static constexpr const char* _type_key = "Load";
  TVM_DECLARE_FINAL_OBJECT_INFO(LoadNode, PrimExprNode);
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
  int lanes;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("base", &base);
    v->Visit("stride", &stride);
    v->Visit("lanes", &lanes);
  }

  bool SEqualReduce(const RampNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(base, other->base) &&
        equal(stride, other->stride) &&
        equal(lanes, other->lanes);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(base);
    hash_reduce(stride);
    hash_reduce(lanes);
  }

  TVM_DLL static PrimExpr make(PrimExpr base, PrimExpr stride, int lanes);

  static constexpr const char* _type_key = "Ramp";
  TVM_DECLARE_FINAL_OBJECT_INFO(RampNode, PrimExprNode);
};

/*! \brief Create a vector where all the elements are value. */
class BroadcastNode : public PrimExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr value;
  /*! \brief The number of lanes. */
  int lanes;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
    v->Visit("lanes", &lanes);
  }

  bool SEqualReduce(const BroadcastNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(value, other->value) &&
        equal(lanes, other->lanes);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
    hash_reduce(lanes);
  }

  TVM_DLL static PrimExpr make(PrimExpr value, int lanes);

  static constexpr const char* _type_key = "Broadcast";
  TVM_DECLARE_FINAL_OBJECT_INFO(BroadcastNode, PrimExprNode);
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  bool SEqualReduce(const LetNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal.DefEqual(var, other->var) &&
        equal(value, other->value) &&
        equal(body, other->body);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce.DefHash(var);
    hash_reduce(value);
    hash_reduce(body);
  }

  TVM_DLL static PrimExpr make(Var var, PrimExpr value, PrimExpr body);

  static constexpr const char* _type_key = "Let";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetNode, PrimExprNode);
};

// Call node, represent a function call or a multi-dimensional array load.
//
// TODO(tvm-team):
// Refactor call with more explicit property registrations.
// rather than calling a string symbol.
// We should move most information into function itself and remove name.

/*! \brief Base node of internal functions. */
class FunctionBaseNode : public Object {
 public:
  /*! \brief virtual destructor */
  virtual ~FunctionBaseNode() {}
  /*! \return the name of the function */
  virtual const std::string& func_name() const = 0;
  /*! \return the number of outputs of this function */
  virtual int num_outputs() const = 0;

  // fall back to pointer equality now before refactor.
  bool SEqualReduce(const FunctionBaseNode* other, SEqualReducer equal) const {
    return this == other;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
};

/*! \brief reference to a function */
class FunctionRef : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FunctionRef, ObjectRef, FunctionBaseNode);
};

/*!
 * \brief Call node.
 */
class CallNode : public PrimExprNode {
 public:
  /*! \brief Possible types of calls. */
  enum CallType : int {
    /*! \brief Extern "C" function. */
    Extern = 0,
    /*! \brief Extern CXX function. */
    ExternCPlusPlus = 1,
    /*! \brief Extern "C" without side-effect. */
    PureExtern = 2,
    /*!
     * \brief Halide-style call, evaluates func(args).
     * \note Deprecated, move to BufferLoad in the future.
     */
    Halide = 3,
    /*! \brief Intrinsic functions. */
    Intrinsic = 4,
    /*! \brief Intrinsic functions that are pure. */
    PureIntrinsic = 5
  };
  /*! \brief The name of the function/intrinsic. */
  std::string name;
  /*! \brief The arguments. */
  Array<PrimExpr> args;
  /*! \brief Type of calls. */
  CallType call_type;
  /*!
   * \brief The function to be called.
   * \note Deprecated, move to BufferLoad in the future.
   */
  FunctionRef func;
  /*!
   * \brief The output value index if func's value is a tuple.
   * \note Deprecated, move to BufferLoad in the future.
   */
  int value_index{0};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("call_type", &call_type);
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
  }

  bool SEqualReduce(const CallNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(name, other->name) &&
        equal(args, other->args) &&
        equal(call_type, other->call_type) &&
        equal(func, other->func) &&
        equal(value_index, other->value_index);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(name);
    hash_reduce(args);
    hash_reduce(call_type);
    hash_reduce(func);
    hash_reduce(value_index);
  }

  TVM_DLL static PrimExpr make(DataType dtype,
                               std::string name,
                               Array<PrimExpr> args,
                               CallType call_type,
                               FunctionRef func = FunctionRef(),
                               int value_index = 0);

  /*! \return Whether call node is pure. */
  bool is_pure() const {
    return (call_type == PureExtern ||
            call_type == PureIntrinsic ||
            call_type == Halide);
  }

  /*!
   * \return Whether call node corresponds to a defined intrinsic.
   * \param intrin_name The name of the intrinsic.
   */
  bool is_intrinsic(const char* intrin_name) const {
    return
        ((call_type == Intrinsic ||
          call_type == PureIntrinsic) &&
         name == intrin_name);
  }

  /*! \return Whether call node can be vectorized. */
  bool is_vectorizable() const;

  static constexpr const char* _type_key = "Call";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, PrimExprNode);

  // Build-in intrinsics
  static constexpr const char* reinterpret = "reinterpret";
  static constexpr const char* bitwise_and = "bitwise_and";
  static constexpr const char* bitwise_not = "bitwise_not";
  static constexpr const char* bitwise_xor = "bitwise_xor";
  static constexpr const char* bitwise_or = "bitwise_or";
  static constexpr const char* shift_left = "shift_left";
  static constexpr const char* shift_right = "shift_right";
  static constexpr const char* popcount = "popcount";
  static constexpr const char* likely = "likely";
  static constexpr const char* glsl_texture_store = "glsl_texture_store";
  static constexpr const char* prefetch = "prefetch";
  static constexpr const char* isnan = "isnan";
  static constexpr const char* isfinite = "isfinite";
  static constexpr const char* isinf = "isinf";

  /*! \brief Vectorizable intrinsic list. */
  static const char* vectorizable_intrinsics[];
};

/*!
 * \brief Shuffle instruction.
 *  vec = concat(vectors)
 *  result = (vec[indices[0]], vec[indices[1]] ...)
 */
class ShuffleNode : public PrimExprNode {
 public:
  /*! \brief the input vectors. */
  Array<PrimExpr> vectors;
  /*! \brief The indices of each element. */
  Array<PrimExpr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("vectors", &vectors);
    v->Visit("indices", &indices);
  }

  bool SEqualReduce(const ShuffleNode* other, SEqualReducer equal) const {
    return
        equal(dtype, other->dtype) &&
        equal(vectors, other->vectors) &&
        equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(vectors);
    hash_reduce(indices);
  }

  TVM_DLL static PrimExpr make(Array<PrimExpr> vectors, Array<PrimExpr> indices);
  TVM_DLL static PrimExpr make_concat(Array<PrimExpr> vectors);
  TVM_DLL static PrimExpr make_extract_element(PrimExpr vector, int index);

  static constexpr const char* _type_key = "Shuffle";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShuffleNode, PrimExprNode);
};

// Reduce operator
class CommReducerNode;

class CommReducer : public ObjectRef {
 public:
  CommReducer() {}
  explicit CommReducer(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const CommReducerNode* get() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const CommReducerNode* operator->() const;
  /*! \brief type indicate the container type */
  using ContainerType = CommReducerNode;
};

/*!
 * \brief A commutative reducer node to represent a commutative
 *  binary operator with identity element
 */
class CommReducerNode : public Object {
 public:
  /*! \brief The left argument of reducer */
  Array<Var> lhs;
  /*! \brief The right argument of reducer */
  Array<Var> rhs;
  /*! \brief The result of reducer */
  Array<PrimExpr> result;
  /*!
   * \brief The identity element of reducer, which leaves other
   *  elements unchanged when combined with it, with respect to
   *  the binary operation of this reducer uses.
   */
  Array<PrimExpr> identity_element;
  /*! \brief Function call operator to combine a and b */
  Array<PrimExpr> operator()(Array<PrimExpr> a, Array<PrimExpr> b) const;
  /*! \brief construct CommReducer from args, result and identity_element */
  TVM_DLL static CommReducer make(Array<Var> lhs,
                                  Array<Var> rhs,
                                  Array<PrimExpr> result,
                                  Array<PrimExpr> identity_element);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("result", &result);
    v->Visit("identity_element", &identity_element);
  }

  bool SEqualReduce(const CommReducerNode* other, SEqualReducer equal) const {
    return
        equal.DefEqual(lhs, other->lhs) &&
        equal.DefEqual(rhs, other->rhs) &&
        equal(result, other->result) &&
        equal(identity_element, other->identity_element);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(lhs);
    hash_reduce.DefHash(rhs);
    hash_reduce(result);
    hash_reduce(identity_element);
  }

  static constexpr const char* _type_key = "CommReducer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(CommReducerNode, Object);
};

inline const CommReducerNode* CommReducer::get() const {
  return static_cast<const CommReducerNode*>(data_.get());
}
inline const CommReducerNode* CommReducer::operator->() const {
  return get();
}

/*! \brief Reduction operator operator */
class ReduceNode : public PrimExprNode {
 public:
  /*! \brief The commutative combiner */
  CommReducer combiner;
  /*! \brief The source operand */
  Array<PrimExpr> source;
  /*! \brief The reduction axis */
  Array<IterVar> axis;
  /*!
   * \brief Predicate on the reduction
   *  Only add the body to reduction if condition is true.
   */
  PrimExpr condition;
  /*! \brief the index of this reduce node */
  int value_index;

  /*! \brief construct expr from op and rdom */
  TVM_DLL static PrimExpr make(CommReducer combiner,
                               Array<PrimExpr> src,
                               Array<IterVar> rdom,
                               PrimExpr condition,
                               int value_index);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("combiner", &combiner);
    v->Visit("source", &source);
    v->Visit("axis", &axis);
    v->Visit("condition", &condition);
    v->Visit("value_index", &value_index);
  }

  bool SEqualReduce(const ReduceNode* other, SEqualReducer equal) const {
    // check axis first so IterVars can define the necessary variables.
    return
        equal(dtype, other->dtype) &&
        equal(axis, other->axis) &&
        equal(combiner, other->combiner) &&
        equal(source, other->source) &&
        equal(condition, other->condition) &&
        equal(value_index, other->value_index);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(axis);
    hash_reduce(combiner);
    hash_reduce(source);
    hash_reduce(condition);
    hash_reduce(value_index);
  }

  static constexpr const char* _type_key = "Reduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReduceNode, PrimExprNode);
};

/*! \brief Any shape. */
class AnyNode : public PrimExprNode {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  bool SEqualReduce(const AnyNode* other, SEqualReducer equal) const {
    return true;
  }

  void SHashReduce(SHashReducer hash_reduce) const {
  }

  /*! \brief Convert to var. */
  Var ToVar() const {
    return Var("any_dim", DataType::Int(32));
  }

  TVM_DLL static PrimExpr make();

  static constexpr const char* _type_key = "Any";
  TVM_DECLARE_FINAL_OBJECT_INFO(AnyNode, PrimExprNode);
};


/*
 * \brief Template function to convert Map to unordered_map
 *  Sometimes useful for API gluing when internal uses unordered_map
 * \param dmap The container map
 * \return The corresponding unordered_map.
 * \tparam K the key of the Map.
 * \tparam V the value of the Map.
 */
template<typename K, typename V>
inline std::unordered_map<K, V> as_unordered_map(const Map<K, V>& dmap) {
  std::unordered_map<K, V> ret;
  for (auto kv : dmap) {
    ret[kv.first] = kv.second;
  }
  return ret;
}

/*! \brief namespace of TVM Intrinsic functions */
namespace intrinsic {
/*!
 * \brief See pesudo code
 *
 *  Construct a big uint that may not be representable by int64
 *
 *  Expr tvm_large_uint_imm(uint32_t v0, uin32_t v1) {
 *    return (v1 << 32) | v0;
 *  }
 */
constexpr const char* tvm_large_uint_imm = "tvm_large_uint_imm";
/*!
 * \brief See pesudo code
 *
 *  Handle tvm_address_of(Load *op) {
 *     return &op->buffer_var[index];
 *  }
 */
constexpr const char* tvm_address_of = "tvm_address_of";
/*!
 * \brief Same as select, used for unsafe memory access.
 *
 *  Type tvm_if_then_else(cond, a, b) {
 *    return cond ? a : b;
 *  }
 */
constexpr const char* tvm_if_then_else = "tvm_if_then_else";
/*!
 * \brief Get head access address with memory access pattern info.
 *
 *  This operator also marks range of the memory access
 *  The offset and extent are in unit of the DType(including vectorization factor).
 *  rw_mask is a bit_mask setting whether the access is a read(1) or write(2).
 *  The access is assume to happen in the current expression.
 *
 *  PtrType tvm_access_ptr(Expr dtype, DType* data,
 *                         int offset, int extent,
 *                         int rw_mask) {
 *    // DType == dtype.type();
 *    return &data[offset];
 *  }
 */
constexpr const char* tvm_access_ptr = "tvm_access_ptr";
/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 *  can be used to cache function local static resources.
 */
constexpr const char* tvm_static_handle = "tvm_static_handle";
/*!
 * \brief Return a unique context id, used for hint of workspace separation.
 *  Different context id ganrantees not having overlapping workspace.
 */
constexpr const char* tvm_context_id = "tvm_context_id";
/*!
 * \brief tvm_tuple is not an actual function and cannot codegen.
 *  It is used to represent tuple structure in value field of AttrStmt,
 *  for the sake of giving hint to optimization.
 *
 *  Handle tvm_tuple(value0, value1, ..., value_n);
 */
constexpr const char* tvm_tuple = "tvm_tuple";
/*!
 * \brief See pesudo code
 *
 *  Type tvm_struct_get(StructType* arr, int index, int field_id) {
 *     return arr[index]->field;
 *  }
 * \sa TVMStructFieldKind
 */
constexpr const char* tvm_struct_get = "tvm_struct_get";
/*!
 * \brief See pesudo code
 *
 *  Handle tvm_struct_set(StructType* arr, int index, int field_id, value) {
 *     arr[index]->field = value;
 *  }
 * \sa TVMStructFieldKind
 */
constexpr const char* tvm_struct_set = "tvm_struct_set";
/*!
 * \brief See pesudo code
 *
 *  bool tvm_handle_is_null(void* handle) {
 *     return handle == nullptr
 *  }
 */
constexpr const char* tvm_handle_is_null = "tvm_handle_is_null";
/*!
 * \brief See pesudo code
 *
 *  void tvm_throw_last_error() {
 *    throw TVMGetLastError();
 *  }
 */
constexpr const char* tvm_throw_last_error = "tvm_throw_last_error";
/*!
 * \brief See pesudo code
 *
 *  dtype in {shape, array, arg_value, arg_tcode}
 *
 *  Handle tvm_stack_alloca(string dtype, int num) {
 *     return new on stack dtype[num];
 *  }
 */
constexpr const char* tvm_stack_alloca = "tvm_stack_alloca";
/*!
 * \brief Allocate a shape tuple on stack, return the handle.
 *
 *  Handle tvm_stack_make_shape(list args) {
 *     ret = alloca stack int64_t[len(args)];
 *     for i in range(len(args)):
 *        ret[i] = args[i]
 *     return &ret[0];
 *  }
 */
constexpr const char* tvm_stack_make_shape = "tvm_stack_make_shape";
/*!
 * \brief Allocate a NDArray(DLTensor) on stack, return the handle.
 *
 *  Type tvm_stack_make_array(Expr data,
 *                            Expr shape,
 *                            Expr strides,
 *                            Expr ndim,
 *                            Expr dtype,
 *                            Expr elem_offset) {
 *     ret = alloca stack DLTensor();
 *     ret->data = data;
 *     ret->shape = shape;
 *     ret->strides = strides != 0 ? strides : nullptr;
 *     ret->ndim = ndim;
 *     ret->dtype = dtype.type();
 *     ret->byte_offset = elem_offset * sizeof(dtype);
 *     return ret;
 *  }
 */
constexpr const char* tvm_stack_make_array = "tvm_stack_make_array";
/*!
 * \brief See pesudo code
 *
 *  int tvm_call_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     return 0;
 *  }
 */
constexpr const char* tvm_call_packed = "tvm_call_packed";
/*!
 * \brief See pesudo code
 *
 *  int tvm_call_trace_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     return 0;
 *  }
 */
constexpr const char *tvm_call_trace_packed = "tvm_call_trace_packed";
/*!
 * \brief See pesudo code
 *  Mark the content as thread local context, can get optimized
 *  by only call the call once at thread start.
 *
 *  Do not allow nesting(getting a thread context from another).
 *
 *  Handle tvm_thread_context(Expr call) {
 *     return call;
 *  }
 */
constexpr const char* tvm_thread_context = "tvm_thread_context";
/*!
 * \brief Lowered version of call packed, the space of value and
 *  type codes are explicitly allocated.
 *
 *  int tvm_call_packed_lowered(name,
 *                              TVMValue* value_stack,
 *                              int* tcode_stack,
 *                              int begin,
 *                              int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
constexpr const char* tvm_call_packed_lowered = "tvm_call_packed_lowered";
/*!
 * \brief Lowered version of trace intrinsic, the space of value and
 *  type codes are explicitly allocated. The return value is the
 *  (end - 1) value on the stack.
 *
 *  int tvm_call_trace_packed_lowered(name,
 *                                    TVMValue* value_stack,
 *                                    int* tcode_stack,
 *                                    int begin,
 *                                    int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
constexpr const char *tvm_call_trace_packed_lowered =
    "tvm_call_trace_packed_lowered";
/*!
 * \brief See pseudo code
 *
 *  int tvm_storage_sync(std::string storage_scope) {
 *     __sync(storage_scope);
 *     return 0;
 *  }
 */
constexpr const char* tvm_storage_sync = "tvm_storage_sync";
/*!
 * \brief See pseudo code
 *
 *  Type tvm_warp_shuffle(Type value, warp_id, width, warp_size) {
 *     return (value passed in by warp indicated by warp_id);
 *  }
 *
 *  Parameter warp_id indicates the source thread ID in a warp.
 *
 *  Parameter width indicates the number of threads involved in one
 *  shuffle. See CUDA document for __shfl.
 *
 *  Parameter warp_size is the size of a warp, which helps a backend
 *  to determine wheter the width paramter is legal.
 */
constexpr const char* tvm_warp_shuffle = "tvm_warp_shuffle";
/*!
 * \brief Initialize the global barrier.
 *  Call this at beginning of kernel that need global barrier.
 */
constexpr const char* tvm_global_barrier_kinit = "tvm_global_barrier_kinit";
/*!
 * \brief See pesudo code
 *
 *  void tvm_thread_allreduce(UIntImm size, Expr source0, ..., Expr cond,
 *                            Var reduce_temp0, .., Var thread_idx1, ...) {
 *     // constraint by the other thread_idx remain the same.
 *     // reduce_temp is used to save intermediate result.
 *     reduce_temp0, ... = reduce(combiner, source0, ..., cond
 *       over [thread_idx1, thread_idx2] passed by any caller)
 *  }
 */
constexpr const char* tvm_thread_allreduce = "tvm_thread_allreduce";
/*!
 * \brief tvm intrinsic for tensor core load operators.
 *
 *  void tvm_load_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                            Expr index, Expr buffer_ptr, Expr stride,
 *                            StringImm layout) {
 *    // m, n, k are the shape of wmma fragment.
 *    // Determine fragment layout(column-major or row major) by layout.
 *    // fragments must be in 'wmma.matrix_a' or 'wmma.matrix_b' scope.
 *    nvcuda::wmma::load_matrix_sync(fragment[index], buffer_ptr, stride);
 *  }
 */
constexpr const char* tvm_load_matrix_sync = "tvm_load_matrix_sync";
/*!
 * \brief tvm intrinsic for tensor core mma_sync operators.
 *
 *  void tvm_mma_sync(Var fragment_d, Expr index_d,
 *                    Var fragment_a, Expr index_a,
 *                    Var fragment_b, Expr index_b,
 *                    Var fragment_c, Expr index_c) {
 *    nvcuda::wmma::mma_sync(fragment_d[index_d], fragment_a[index_a],
 *                           fragment_b[index_b], fragment_c[index_c]);
 *  }
 */
constexpr const char* tvm_mma_sync = "tvm_mma_sync";
/*!
 * \brief tvm intrinsic for tensor core bmma_sync operators.
 *
 *  void tvm_bmma_sync(Var fragment_d, Expr index_d,
 *                     Var fragment_a, Expr index_a,
 *                     Var fragment_b, Expr index_b,
 *                     Var fragment_c, Expr index_c) {
 *    nvcuda::wmma::bmma_sync(fragment_d[index_d], fragment_a[index_a],
 *                           fragment_b[index_b], fragment_c[index_c]);
 *  }
 */
constexpr const char* tvm_bmma_sync = "tvm_bmma_sync";
/*!
 * \brief tvm intrinsic for tensor core fill_fragment operators.
 *
 *  void tvm_fill_fragment(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                         Expr index, Expr value) {
 *    // m, n, k are the shape of wmma fragment
 *    // fragments must be in 'wmma.accumulator' scope.
 *    nvcuda::wmma::fill_fragment(fragment[index], value);
 *  }
 */
constexpr const char* tvm_fill_fragment = "tvm_fill_fragment";
/*!
 * \brief tvm intrinsic for tensor core store operators.
 *
 *  void tvm_store_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                             Expr index, Expr buffer_ptr, Expr stride,
 *                             StringImm layout) {
 *    // m, n, k are the shape of wmma fragment
 *    // fragments must be in 'wmma.accumulator' scope.
 *    nvcuda::wmma::store_matrix_sync(fragment[index], buffer_ptr, stride, layout);
 *  }
 */
constexpr const char* tvm_store_matrix_sync = "tvm_store_matrix_sync";

/*! \brief The kind of structure field info used in intrinsic */
enum TVMStructFieldKind : int {
  // array head address
  kArrAddr,
  kArrData,
  kArrShape,
  kArrStrides,
  kArrNDim,
  kArrTypeCode,
  kArrTypeBits,
  kArrTypeLanes,
  kArrByteOffset,
  kArrDeviceId,
  kArrDeviceType,
  kArrKindBound_,
  // TVMValue field
  kTVMValueContent,
  kTVMValueKindBound_
};
}   // namespace intrinsic

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace runtime {
// Additional implementattion overloads for PackedFunc.

template<>
struct PackedFuncValueConverter<tvm::Integer> {
  // common rule for RetValue and ArgValue
  static tvm::Integer From(const TVMPODValue_& val) {
    if (val.type_code() == kTVMNullptr) {
      return Integer(ObjectPtr<Object>(nullptr));
    }
    if (val.type_code() == kDLInt) {
      return Integer(val.operator int());
    }
    return val.AsObjectRef<tvm::Integer>();
  }
};
}  // namespace runtime
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::tir::IterVar> : public ::tvm::ObjectHash {
};
}
#endif  // TVM_TIR_EXPR_H_
