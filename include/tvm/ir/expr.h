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
 * \file tvm/ir/expr.h
 * \brief Base expr nodes in TVM.
 */
#ifndef TVM_IR_EXPR_H_
#define TVM_IR_EXPR_H_

#include <tvm/runtime/object.h>
#include <tvm/node/node.h>
#include <tvm/node/container.h>
#include <tvm/ir/span.h>
#include <tvm/ir/type.h>
#include <string>
#include <algorithm>
#include <limits>

namespace tvm {

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class BaseExprNode : public Object {
 public:
  static constexpr const char* _type_key = "Expr";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseExprNode, Object);
};

/*!
 * \brief Managed reference to BaseExprNode.
 * \sa BaseExprNode
 */
class BaseExpr : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseExpr, ObjectRef, BaseExprNode);
};

/*!
 * \brief Base node of all primitive expressions.
 *
 *  A primitive expression deals with low-level
 *  POD data types and handles without
 *  doing life-cycle management for objects.
 *
 *  PrimExpr is used in the low-level code
 *  optimizations and integer analysis.
 *
 * \sa PrimExpr
 */
class PrimExprNode : public BaseExprNode {
 public:
  /*!
   * \brief The runtime data type of the primitive expression.
   *
   * runtime::DataType(dtype) provides coarse grained type information
   * during compile time and runtime. It is eagerly built in
   * PrimExpr expression construction and can be used for
   * quick type checking.
   *
   * dtype is sufficient to decide the Type of the PrimExpr
   * when it corresponds to POD value types such as i32.
   *
   * When dtype is DataType::Handle(), the expression could corresponds to
   * a more fine-grained Type, and we can get the type by running lazy type inference.
   */
  DataType dtype;

  static constexpr const char* _type_key = "PrimExpr";
  TVM_DECLARE_BASE_OBJECT_INFO(PrimExprNode, BaseExprNode);
};

/*!
 * \brief Reference to PrimExprNode.
 * \sa PrimExprNode
 */
class PrimExpr : public BaseExpr {
 public:
  /*!
   * \brief construct from integer.
   * \param value The value to be constructed.
   */
  TVM_DLL PrimExpr(int32_t value);  // NOLINT(*)
  /*!
   * \brief construct from float.
   * \param value The value to be constructed.
   */
  TVM_DLL PrimExpr(float value);  // NOLINT(*)
  /*!
   * \brief construct from string.
   * \param str The value to be constructed.
   */
  TVM_DLL PrimExpr(std::string str);  // NOLINT(*)

  /*! \return the data type of this expression. */
  DataType dtype() const {
    return static_cast<const PrimExprNode*>(get())->dtype;
  }

  TVM_DEFINE_OBJECT_REF_METHODS(PrimExpr, BaseExpr, PrimExprNode);

 private:
  // Internal function for conversion.
  friend class runtime::TVMPODValue_;
  TVM_DLL static PrimExpr FromObject_(ObjectPtr<Object> ptr);
};

/*!
 * \brief Base node of all non-primitive expressions.
 *
 * RelayExpr supports tensor types, functions and ADT as
 * first class citizens. The life-cycle of the corresponding
 * objects are implicitly managed by the language.
 *
 * \sa RelayExpr
 */
class RelayExprNode : public BaseExprNode {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;
  /*!
   * \brief Stores the result of type inference(type checking).
   *
   * \note This can be undefined before type inference.
   *       This value is discarded during serialization.
   */
  mutable Type checked_type_ = Type(nullptr);
  /*!
   * \return The checked_type
   */
  const Type& checked_type() const;
  /*!
   * \brief Check if the inferred(checked) type of the Expr
   *  is backed by a TTypeNode and return it.
   *
   * \note This function will thrown an error if the node type
   *       of this Expr is not TTypeNode.
   *
   * \return The corresponding TTypeNode pointer.
   * \tparam The specific TypeNode we look for.
   */
  template<typename TTypeNode>
  inline const TTypeNode* type_as() const;

  static constexpr const char* _type_key = "relay.Expr";
  TVM_DECLARE_BASE_OBJECT_INFO(RelayExprNode, BaseExprNode);
};

/*!
 * \brief Managed reference to RelayExprNode.
 * \sa RelayExprNode
 */
class RelayExpr : public BaseExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RelayExpr, BaseExpr, RelayExprNode);
};

class GlobalVar;
/*!
 * \brief Global variable that leaves in the top-level module.
 *
 * A GlobalVar only refers to function definitions.
 * This is used to enable recursive calls between function.
 *
 * \sa GlobalVarNode
 */
class GlobalVarNode : public RelayExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  std::string name_hint;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  static constexpr const char* _type_key = "relay.GlobalVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(GlobalVarNode, RelayExprNode);
};

/*!
 * \brief Managed reference to GlobalVarNode.
 * \sa GlobalVarNode
 */
class GlobalVar : public RelayExpr {
 public:
  TVM_DLL explicit GlobalVar(std::string name_hint);

  TVM_DEFINE_OBJECT_REF_METHODS(GlobalVar, RelayExpr, GlobalVarNode);
};

/*!
 * \brief Base node of all functions.
 *
 * We support several variants of functions throughout the stack.
 * All of the functions shares the same type system(via checked_type)
 * to support cross variant calls.
 *
 * \sa BaseFunc
 */
class BaseFuncNode : public RelayExprNode {
 public:
  static constexpr const char* _type_key = "BaseFunc";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseFuncNode, RelayExprNode);
};

/*!
 * \brief Managed reference to BaseFuncNode.
 * \sa BaseFuncNode
 */
class BaseFunc : public RelayExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseFunc, RelayExpr, BaseFuncNode);
};

// PrimExprs that are useful as runtime containers.
//
/*!
 * \brief Constant integer literals in the program.
 * \sa IntImm
 */
class IntImmNode : public PrimExprNode {
 public:
  /*! \brief the Internal value. */
  int64_t value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "IntImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(IntImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference class to IntImmNode.
 *
 * \sa IntImmNode
 */
class IntImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param dtype The data type of the value.
   * \param value The internal value.
   */
  TVM_DLL IntImm(DataType dtype, int64_t value);

  TVM_DEFINE_OBJECT_REF_METHODS(IntImm, PrimExpr, IntImmNode);
};

/*!
 * \brief Constant floating point literals in the program.
 * \sa FloatImm
 */
class FloatImmNode : public PrimExprNode {
 public:
  /*! \brief The constant value content. */
  double value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "FloatImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(FloatImmNode, PrimExprNode);
};

/*!
 * \brief Managed reference class to FloatImmNode.
 *
 * \sa FloatImmNode
 */
class FloatImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param dtype The data type of the value.
   * \param value The internal value.
   */
  TVM_DLL FloatImm(DataType dtype, double value);

  TVM_DEFINE_OBJECT_REF_METHODS(FloatImm, PrimExpr, FloatImmNode);
};

/*!
 * \brief Container of constant int that adds more constructors.
 *
 * This is used to store and automate type check
 * attributes that must be constant integer.
 *
 * \sa IntImm
 */
class Integer : public IntImm {
 public:
  Integer() {}
  /*!
   * \brief constructor from node.
   */
  explicit Integer(ObjectPtr<Object> node) : IntImm(node) {}
  /*!
   * \brief Construct integer from int value.
   */
  Integer(int value) : IntImm(DataType::Int(32), value) {}  // NOLINT(*)
  /*!
   * \brief Construct integer from int imm.
   * \param other The other value.
   */
  Integer(IntImm other) : IntImm(std::move(other)) {}  // NOLINT(*)
  /*!
   * \brief Assign an expression to integer.
   * \param other another expression.
   */
  Integer& operator=(const IntImm& other) {
    data_ = ObjectRef::GetDataPtr<Object>(other);
    return *this;
  }
  /*!
   * \brief convert to int64_t
   */
  operator int64_t() const {
    CHECK(data_ != nullptr)
        << " Trying to reference a null Integer";
    return (*this)->value;
  }
};

/*! \brief range over one dimension */
class RangeNode : public Object {
 public:
  /*! \brief beginning of the node */
  PrimExpr min;
  /*! \brief the extend of range */
  PrimExpr extent;
  /*! \brief constructor */
  RangeNode() {}
  RangeNode(PrimExpr min, PrimExpr extent) : min(min), extent(extent) {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("min", &min);
    v->Visit("extent", &extent);
  }

  static constexpr const char* _type_key = "Range";
  TVM_DECLARE_FINAL_OBJECT_INFO(RangeNode, Object);
};

/*! \brief Range constainer  */
class Range : public ObjectRef {
 public:
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   */
  TVM_DLL Range(PrimExpr begin, PrimExpr end);
  /*!
   * \brief construct a new range with min and extent
   *  The corresponding constructor is removed,
   *  because that is counter convention of tradition meaning
   *  of range(begin, end)
   *
   * \param min The minimum range.
   * \param extent The extent of the range.
   */
  static Range make_by_min_extent(PrimExpr min, PrimExpr extent);
  // declare range.
  TVM_DEFINE_OBJECT_REF_METHODS(Range, ObjectRef, RangeNode);
};

// implementataions
inline const Type& RelayExprNode::checked_type() const {
  CHECK(checked_type_.defined())
      << "internal error: the type checker has "
      << "not populated the checked_type "
      << "field for "
      << GetRef<RelayExpr>(this);
  return this->checked_type_;
}

template<typename TTypeNode>
inline const TTypeNode* RelayExprNode::type_as() const {
  static_assert(std::is_base_of<TypeNode, TTypeNode>::value,
                "TType must be a special case of type");
  CHECK(checked_type_.defined())
      << "Type inference for this Expr has not completed. Try to call infer_type pass.";
  const TTypeNode* node = checked_type_.as<TTypeNode>();
  CHECK(node != nullptr)
      << "Expected type to be " << TTypeNode::_type_key
      << ", but get " << checked_type_->GetTypeKey();
  return node;
}

}  // namespace tvm

namespace tvm {
namespace runtime {
// Additional implementattion overloads for PackedFunc.
inline TVMPODValue_::operator tvm::PrimExpr() const {
  if (type_code_ == kTVMNullptr) return PrimExpr();
  if (type_code_ == kDLInt) {
    CHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    CHECK_GE(value_.v_int64, std::numeric_limits<int>::min());
    return PrimExpr(static_cast<int>(value_.v_int64));
  }
  if (type_code_ == kDLFloat) {
    return PrimExpr(static_cast<float>(value_.v_float64));
  }

  TVM_CHECK_TYPE_CODE(type_code_, kTVMObjectHandle);
  Object* ptr = static_cast<Object*>(value_.v_handle);
  return PrimExpr::FromObject_(ObjectPtr<Object>(ptr));
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_IR_EXPR_H_
