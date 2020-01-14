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
  /*! \brief Cosntructor */
  BaseExpr() {}
  /*!
   * \brief Cosntructor from object ptr.
   * \param ptr The object pointer.
   */
  explicit BaseExpr(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  /*! \brief The container type. */
  using ContainerType = BaseExprNode;
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
    /*! \brief Cosntructor */
  PrimExpr() {}
  /*!
   * \brief Cosntructor from object ptr.
   * \param ptr The object pointer.
   */
  explicit PrimExpr(ObjectPtr<Object> ptr) : BaseExpr(ptr) {}
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
  /*! \brief The container type. */
  using ContainerType = PrimExprNode;
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
#endif  // TVM_IR_EXPR_H_
