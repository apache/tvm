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
 * \file tvm/ir/adt.h
 * \brief Algebraic data type definitions.
 *
 * We adopt relay's ADT definition as a unified class
 * for decripting structured data.
 */
#ifndef TVM_IR_ADT_H_
#define TVM_IR_ADT_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/node/node.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include <string>

namespace tvm {

/*!
 * \brief ADT constructor.
 * Constructors compare by pointer equality.
 * \sa Constructor
 */
class ConstructorNode : public RelayExprNode {
 public:
  /*! \brief The name (only a hint) */
  String name_hint;
  /*! \brief Input to the constructor. */
  Array<Type> inputs;
  /*! \brief The datatype the constructor will construct. */
  GlobalTypeVar belong_to;
  /*! \brief Index in the table of constructors (set when the type is registered). */
  mutable int32_t tag = -1;

  ConstructorNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("inputs", &inputs);
    v->Visit("belong_to", &belong_to);
    v->Visit("tag", &tag);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const ConstructorNode* other, SEqualReducer equal) const {
    // Use namehint for now to be consistent with the legacy relay impl
    // TODO(tvm-team) revisit, need to check the type var.
    return equal(name_hint, other->name_hint) && equal(inputs, other->inputs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name_hint);
    hash_reduce(inputs);
  }

  static constexpr const char* _type_key = "relay.Constructor";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstructorNode, RelayExprNode);
};

/*!
 * \brief Managed reference to ConstructorNode
 * \sa ConstructorNode
 */
class Constructor : public RelayExpr {
 public:
  /*!
   * \brief Constructor
   * \param name_hint the name of the constructor.
   * \param inputs The input types.
   * \param belong_to The data type var the constructor will construct.
   */
  TVM_DLL Constructor(String name_hint, Array<Type> inputs, GlobalTypeVar belong_to);

  TVM_DEFINE_OBJECT_REF_METHODS(Constructor, RelayExpr, ConstructorNode);
};

/*! \brief TypeData container node */
class TypeDataNode : public TypeNode {
 public:
  /*!
   * \brief The header is simply the name of the ADT.
   * We adopt nominal typing for ADT definitions;
   * that is, differently-named ADT definitions with same constructors
   * have different types.
   */
  GlobalTypeVar header;
  /*! \brief The type variables (to allow for polymorphism). */
  Array<TypeVar> type_vars;
  /*! \brief The constructors. */
  Array<Constructor> constructors;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("header", &header);
    v->Visit("type_vars", &type_vars);
    v->Visit("constructors", &constructors);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const TypeDataNode* other, SEqualReducer equal) const {
    return equal.DefEqual(header, other->header) && equal.DefEqual(type_vars, other->type_vars) &&
           equal(constructors, other->constructors);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(header);
    hash_reduce.DefHash(type_vars);
    hash_reduce(constructors);
  }

  static constexpr const char* _type_key = "relay.TypeData";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeDataNode, TypeNode);
};

/*!
 * \brief Stores all data for an Algebraic Data Type (ADT).
 *
 * In particular, it stores the handle (global type var) for an ADT
 * and the constructors used to build it and is kept in the module. Note
 * that type parameters are also indicated in the type data: this means that
 * for any instance of an ADT, the type parameters must be indicated. That is,
 * an ADT definition is treated as a type-level function, so an ADT handle
 * must be wrapped in a TypeCall node that instantiates the type-level arguments.
 * The kind checker enforces this.
 */
class TypeData : public Type {
 public:
  /*!
   * \brief Constructor
   * \param header the name of ADT.
   * \param type_vars type variables.
   * \param constructors constructors field.
   */
  TVM_DLL TypeData(GlobalTypeVar header, Array<TypeVar> type_vars, Array<Constructor> constructors);

  TVM_DEFINE_OBJECT_REF_METHODS(TypeData, Type, TypeDataNode);
};

}  // namespace tvm
#endif  // TVM_IR_ADT_H_
