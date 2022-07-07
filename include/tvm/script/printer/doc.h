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
#ifndef TVM_SCRIPT_PRINTER_DOC_H_
#define TVM_SCRIPT_PRINTER_DOC_H_

#include <tvm/ir/expr.h>
#include <tvm/node/node.h>
#include <tvm/runtime/data_type.h>

namespace tvm {
namespace script {
namespace printer {

/*!
 * \brief The base class of all Doc.
 *
 * Doc is an intermediate representation between IR from TVM
 * and the TVMScript code.
 * During printing, IR graph is first translated into Doc tree,
 * then the Doc tree is translated to the target language in
 * text format.
 *
 * \sa Doc
 */
class DocNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "script.printer.Doc";
  TVM_DECLARE_BASE_OBJECT_INFO(DocNode, Object);

 public:
  virtual ~DocNode() = default;
};

/*!
 * \brief Reference type of DocNode.
 *
 * \sa DocNode
 */
class Doc : public ObjectRef {
 protected:
  Doc() = default;

 public:
  virtual ~Doc() = default;
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Doc, ObjectRef, DocNode);
};

/*!
 * \brief The base class of expression doc.
 *
 * \sa ExprDoc
 */
class ExprDocNode : public DocNode {
 public:
  void VisitAttrs(AttrVisitor* v) { DocNode::VisitAttrs(v); }

  static constexpr const char* _type_key = "script.printer.ExprDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprDocNode, DocNode);
};

/*!
 * \brief Reference type of ExprDocNode.
 *
 * \sa ExprDocNode
 */
class ExprDoc : public Doc {
 protected:
  ExprDoc() = default;

 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExprDoc, Doc, ExprDocNode);
};

/*!
 * \brief Doc that represents literal value.
 *
 * \sa LiteralDoc
 */
class LiteralDocNode : public ExprDocNode {
 public:
  /*!
   * \brief the internal representation of the literal value.
   *
   * Possible actual types:
   * - IntImm (integer or boolean)
   * - FloatImm
   * - String
   * - null
   */
  ObjectRef value;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "script.printer.LiteralDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LiteralDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of LiteralDocNode.
 *
 * \sa LiteralDocNode
 */
class LiteralDoc : public ExprDoc {
 protected:
  explicit LiteralDoc(ObjectRef value);

 public:
  /*!
   * \brief Create a LiteralDoc to represent None/null/empty value.
   */
  static LiteralDoc None() { return LiteralDoc(ObjectRef(nullptr)); }

  /*!
   * \brief Create a LiteralDoc to represent integer.
   * \param v The integer value.
   */
  static LiteralDoc Int(int v) { return LiteralDoc(IntImm(DataType::Int(64), v)); }

  /*!
   * \brief Create a LiteralDoc to represent boolean.
   * \param v The boolean value.
   */
  static LiteralDoc Boolean(bool v) { return LiteralDoc(IntImm(DataType::Bool(), v)); }

  /*!
   * \brief Create a LiteralDoc to represent float.
   * \param v The float value.
   */
  static LiteralDoc Float(double v) { return LiteralDoc(FloatImm(DataType::Float(64), v)); }

  /*!
   * \brief Create a LiteralDoc to represent string.
   * \param v The string value.
   */
  static LiteralDoc Str(const String& v) { return LiteralDoc(v); }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LiteralDoc, ExprDoc, LiteralDocNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_H_
