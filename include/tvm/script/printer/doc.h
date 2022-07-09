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

class ExprDoc;

/*!
 * \brief The base class of expression doc.
 *
 * \sa ExprDoc
 */
class ExprDocNode : public DocNode {
 public:
  /*!
   * \brief Create a doc representing attribute access on the current ExprDoc
   * \param attr The attribute to access.
   */
  ExprDoc Attr(String attr) const;

  /*!
   * \brief Create a doc representing index access on the current ExprDoc
   * \param indices The indices to access.
   */
  ExprDoc Index(Array<Doc> indices) const;

  /*!
   * \brief Create a doc representing calling the current ExprDoc
   * \param args The positional arguments of the function call.
   */
  ExprDoc Call(Array<ExprDoc, void> args) const;

  /*!
   * \brief Create a doc representing attribute access on the current ExprDoc
   * \param args The positional arguments of the function call.
   * \param kwargs_keys Keys of keywords arguments of the function call.
   * \param kwargs_values Values of keywords arguments of the function call.
   */
  ExprDoc Call(Array<ExprDoc, void> args,        //
               Array<String, void> kwargs_keys,  //
               Array<ExprDoc, void> kwargs_values) const;

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

/*!
 * \brief Doc that represents identifier.
 *
 * \sa IdDoc
 */
class IdDocNode : public ExprDocNode {
 public:
  /*! \brief The name of the identifier */
  String name;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "script.printer.IdDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of IdDocNode.
 *
 * \sa IdDocNode
 */
class IdDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of IdDoc.
   * \param name The name of identifier.
   */
  explicit IdDoc(String name);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(IdDoc, ExprDoc, IdDocNode);
};

/*!
 * \brief Doc that represents attribute access on another expression.
 *
 * \sa AttrAccessDoc
 */
class AttrAccessDocNode : public ExprDocNode {
 public:
  /*! \brief The target expression to be accessed */
  ExprDoc value{nullptr};
  /*! \brief The attribute to be accessed */
  String attr;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("attr", &attr);
  }

  static constexpr const char* _type_key = "script.printer.AttrAccessDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrAccessDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of AttrAccessDocNode.
 *
 * \sa AttrAccessDocNode
 */
class AttrAccessDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of AttrAccessDoc
   * \param value The target expression of attribute access.
   * \param attr The name of attribute to access.
   */
  explicit AttrAccessDoc(ExprDoc value, String attr);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AttrAccessDoc, ExprDoc, AttrAccessDocNode);
};

/*!
 * \brief Doc that represents index access on another expression.
 *
 * \sa IndexDoc
 */
class IndexDocNode : public ExprDocNode {
 public:
  /*! \brief The container value to be accessed */
  ExprDoc value{nullptr};
  /*!
   * \brief The indices to access
   *
   * Possible actual types:
   * - ExprDoc (single point access like a[1, 2])
   * - SliceDoc (slice access like a[1:5, 2])
   */
  Array<Doc> indices;  // Each element is union of: Slice / ExprDoc

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
  }

  static constexpr const char* _type_key = "script.printer.IndexDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndexDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of IndexDocNode.
 *
 * \sa IndexDocNode
 */
class IndexDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of IndexDoc
   * \param value The target expression of index access.
   * \param indices The indices to access.
   */
  explicit IndexDoc(ExprDoc value, Array<Doc> indices);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(IndexDoc, ExprDoc, IndexDocNode);
};

/*!
 * \brief Doc that represents function call.
 *
 * \sa CallDoc
 */
class CallDocNode : public ExprDocNode {
 public:
  /*! \brief The callee of this function call */
  ExprDoc callee{nullptr};
  /*! \brief The positional arguments */
  Array<ExprDoc> args;
  /*! \brief The keys of keyword arguments */
  Array<String> kwargs_keys;
  /*!
   * \brief The values of keyword arguments.
   *
   * The i-th element is the value of the i-th key in `kwargs_keys`.
   * It must have the same length as `kwargs_keys`.
   */
  Array<ExprDoc> kwargs_values;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("callee", &callee);
    v->Visit("args", &args);
    v->Visit("kwargs_keys", &kwargs_keys);
    v->Visit("kwargs_values", &kwargs_values);
  }

  static constexpr const char* _type_key = "script.printer.CallDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of CallDocNode.
 *
 * \sa CallDocNode
 */
class CallDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of CallDoc
   * \param callee The callee of this function call.
   * \param args The positional arguments.
   * \param kwargs_keys Keys of keyword arguments.
   * \param kwargs_values Values of keyword arguments, must have the same length as `kwargs_keys.
   */
  CallDoc(ExprDoc callee, Array<ExprDoc> args, Array<String> kwargs_keys,
          Array<ExprDoc> kwargs_values);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CallDoc, ExprDoc, CallDocNode);
};

/*!
 * \brief Doc that represents operation.
 *
 * It can be unary, binary and other special operators (for example,
 * the if-then-else expression).
 *
 * \sa OperationDoc
 */
class OperationDocNode : public ExprDocNode {
 public:
  enum class Kind : int32_t {
    // Unary operators
    kUnaryStart,
    kUSub,    // -x
    kInvert,  // ~x
    kUnaryEnd,

    // Binary operators
    kBinaryStart,
    kAdd,       // +
    kSub,       // -
    kMult,      // *
    kDiv,       // /
    kFloorDiv,  // // in Python
    kMod,       // % in Python
    kPow,       // ** in Python
    kLShift,    // <<
    kRShift,    // >>
    kBitAnd,    // &
    kBitOr,     // |
    kBitXor,    // ^
    kLt,        // <
    kLtE,       // <=
    kEq,        // ==
    kNotEq,     // !=
    kGt,        // >
    kGtE,       // >=
    kBinaryEnd,

    // Special
    kSpecialStart,
    kAssert,
  };

  /*! \brief The kind of operation (operator) */
  Kind kind;
  /*! \brief Operands of this expression */
  Array<ExprDoc> operands;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("kind", &kind);
    v->Visit("operands", &operands);
  }

  static constexpr const char* _type_key = "script.printer.OperationDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(OperationDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of OperationDocNode.
 *
 * \sa OperationDocNode
 */
class OperationDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of OperationDoc
   * \param kind The kind of operation.
   * \param operands Operands of this expression.
   */
  explicit OperationDoc(OperationDocNode::Kind kind, Array<ExprDoc> operands);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(OperationDoc, ExprDoc, OperationDocNode);
};

/*!
 * \brief Doc that represents anonymous function.
 *
 * LambdaDoc can only have positional arguments without type annotation,
 * and a single expression as body.
 *
 * \sa LambdaDoc
 */
class LambdaDocNode : public ExprDocNode {
 public:
  /*! \brief The arguments of this anonymous function */
  Array<IdDoc> args;
  /*! \brief The body of this anonymous function */
  ExprDoc body{nullptr};

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("args", &args);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "script.printer.LambdaDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LambdaDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of LambdaDocNode.
 *
 * \sa LambdaDocNode
 */
class LambdaDoc : public ExprDoc {
 public:
  /*!
   * \brief Constructor of LambdaDoc
   * \param args Arguments of this function.
   * \param body Body expression of this function.
   */
  explicit LambdaDoc(Array<IdDoc> args, ExprDoc body);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LambdaDoc, ExprDoc, LambdaDocNode);
};

/*!
 * \brief Doc that represents tuple literal.
 *
 * \sa TupleDoc
 */
class TupleDocNode : public ExprDocNode {
 public:
  /*! \brief Elements of tuple */
  Array<ExprDoc> elements;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("elements", &elements);
  }

  static constexpr const char* _type_key = "script.printer.TupleDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of TupleDocNode.
 *
 * \sa TupleDocNode
 */
class TupleDoc : public ExprDoc {
 public:
  /*!
   * \brief Create an empty TupleDoc
   */
  TupleDoc() : TupleDoc(runtime::make_object<TupleDocNode>()) {}
  /*!
   * \brief Constructor of TupleDoc
   * \param elements Elements of tuple.
   */
  explicit TupleDoc(Array<ExprDoc> elements);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TupleDoc, ExprDoc, TupleDocNode);
};

/*!
 * \brief Doc that represents list literal.
 *
 * \sa AttrAccessDoc
 */
class ListDocNode : public ExprDocNode {
 public:
  /*! \brief Elements of list */
  Array<ExprDoc> elements;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("elements", &elements);
  }

  static constexpr const char* _type_key = "script.printer.ListDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ListDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of ListDocNode.
 *
 * \sa ListDocNode
 */
class ListDoc : public ExprDoc {
 public:
  /*!
   * \brief Create an empty ListDoc
   */
  ListDoc() : ListDoc(runtime::make_object<ListDocNode>()) {}
  /*!
   * \brief Constructor of ListDoc
   * \param elements Elements of list.
   */
  explicit ListDoc(Array<ExprDoc> elements);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ListDoc, ExprDoc, ListDocNode);
};

/*!
 * \brief Doc that represents dictionary literal.
 *
 * \sa AttrAccessDoc
 */
class DictDocNode : public ExprDocNode {
 public:
  /*! \brief keys of dictionary */
  Array<ExprDoc> keys;
  /*!
   * \brief Values of dictionary
   *
   * The i-th element is the value of the i-th element of `keys`.
   * It must have the same length as `keys`.
   */
  Array<ExprDoc> values;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("keys", &keys);
    v->Visit("values", &values);
  }

  static constexpr const char* _type_key = "script.printer.DictDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(DictDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of DictDocNode.
 *
 * \sa DictDocNode
 */
class DictDoc : public ExprDoc {
 public:
  /*!
   * \brief Create an empty dictionary
   */
  DictDoc() : DictDoc(runtime::make_object<DictDocNode>()) {}
  /*!
   * \brief Constructor of DictDoc
   * \param keys Keys of dictionary.
   * \param values Values of dictionary, must have same length as `keys`.
   */
  explicit DictDoc(Array<ExprDoc> keys, Array<ExprDoc> values);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DictDoc, ExprDoc, DictDocNode);
};

/*!
 * \brief Doc that represents slice in Index expression.
 *
 * This doc can only appear in IndexDoc::indices.
 *
 * \sa AttrAccessDoc
 */
class SliceDocNode : public DocNode {
 public:
  /*! \brief The start of slice */
  Optional<ExprDoc> start;
  /*! \brief The exclusive end of slice */
  Optional<ExprDoc> stop;

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("start", &start);
    v->Visit("stop", &stop);
  }

  static constexpr const char* _type_key = "script.printer.SliceDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(SliceDocNode, DocNode);
};

/*!
 * \brief Reference type of SliceDocNode.
 *
 * \sa SliceDocNode
 */
class SliceDoc : public Doc {
 public:
  /*!
   * \brief Constructor of SliceDoc
   * \param start The start of slice.
   * \param start The exclusive end of slice.
   */
  explicit SliceDoc(Optional<ExprDoc> start, Optional<ExprDoc> stop);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SliceDoc, Doc, SliceDocNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_H_
