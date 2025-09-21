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

#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/node/node.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>

#include <string>

namespace tvm {
namespace script {
namespace printer {

using AccessPath = ffi::reflection::AccessPath;

// Forward declaration
class Doc;

/*!
 * \brief Convert Doc into Python script.
 * \param doc Doc to be converted
 * \param cfg The configuration of the printer
 */
ffi::String DocToPythonScript(Doc doc, const PrinterConfig& cfg);

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
  /*!
   * \brief The list of object paths of the source IR node.
   *
   * This is used to trace back to the IR node position where
   * this Doc is generated, in order to position the diagnostic
   * message.
   */
  mutable ffi::Array<ffi::reflection::AccessPath> source_paths;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DocNode>().def_rw("source_paths", &DocNode::source_paths);
  }

  static constexpr bool _type_mutable = true;

  TVM_FFI_DECLARE_OBJECT_INFO("script.printer.Doc", DocNode, Object);

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
  explicit Doc(ObjectPtr<DocNode> data) : ObjectRef(data) {}

 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Doc, ObjectRef, DocNode);
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
  ExprDoc Attr(ffi::String attr) const;

  /*!
   * \brief Create a doc representing index access on the current ExprDoc
   * \param indices The indices to access.
   */
  ExprDoc operator[](ffi::Array<Doc> indices) const;

  /*!
   * \brief Create a doc representing calling the current ExprDoc
   * \param args The positional arguments of the function call.
   */
  ExprDoc Call(ffi::Array<ExprDoc, void> args) const;

  /*!
   * \brief Create a doc representing attribute access on the current ExprDoc
   * \param args The positional arguments of the function call.
   * \param kwargs_keys Keys of keywords arguments of the function call.
   * \param kwargs_values Values of keywords arguments of the function call.
   */
  ExprDoc Call(ffi::Array<ExprDoc, void> args,       //
               ffi::Array<ffi::String> kwargs_keys,  //
               ffi::Array<ExprDoc, void> kwargs_values) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExprDocNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.printer.ExprDoc", ExprDocNode, DocNode);
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
  /*!
   * \brief Create a doc representing index access on the current ExprDoc
   * \param indices The indices to access.
   */
  ExprDoc operator[](ffi::Array<Doc> indices) const;

  explicit ExprDoc(ObjectPtr<ExprDocNode> data) : Doc(data) { TVM_FFI_ICHECK(data != nullptr); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprDoc, Doc, ExprDocNode);
};

/*!
 * \brief The base class of statement doc.
 *
 * \sa StmtDoc
 */
class StmtDocNode : public DocNode {
 public:
  /*!
   * \brief The comment of this doc.
   *
   * The actual position of the comment depends on the type of Doc
   * and also the DocPrinter implementation. It could be on the same
   * line as the statement, or the line above, or inside the statement
   * if it spans over multiple lines.
   * */
  mutable ffi::Optional<ffi::String> comment{std::nullopt};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StmtDocNode>().def_rw("comment", &StmtDocNode::comment);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.printer.StmtDoc", StmtDocNode, DocNode);
};

/*!
 * \brief Reference type of StmtDocNode.
 *
 * \sa StmtDocNode
 */
class StmtDoc : public Doc {
 protected:
  StmtDoc() = default;

 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtDoc, Doc, StmtDocNode);
};

/*!
 * \brief The container doc that holds a list of StmtDoc.
 * \note `StmtBlockDoc` is never used in the IR, but a temporary container that allows holding a
 * list of StmtDoc.
 * \sa StmtBlockDoc
 */
class StmtBlockDocNode : public DocNode {
 public:
  /*! \brief The list of statements. */
  ffi::Array<StmtDoc> stmts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StmtBlockDocNode>().def_ro("stmts", &StmtBlockDocNode::stmts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.StmtBlockDoc", StmtBlockDocNode, DocNode);
};

/*!
 * \brief Reference type of StmtBlockDocNode.
 * \sa StmtBlockDocNode
 */
class StmtBlockDoc : public Doc {
 public:
  /*!
   * \brief Constructor of StmtBlockDoc.
   * \param stmts The list of statements.
   */
  explicit StmtBlockDoc(ffi::Array<StmtDoc> stmts);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtBlockDoc, Doc, StmtBlockDocNode);
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
  ffi::Any value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LiteralDocNode>().def_ro("value", &LiteralDocNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.LiteralDoc", LiteralDocNode, ExprDocNode);
};

/*!
 * \brief Reference type of LiteralDocNode.
 *
 * \sa LiteralDocNode
 */
class LiteralDoc : public ExprDoc {
 protected:
  explicit LiteralDoc(ffi::Any value, const ffi::Optional<AccessPath>& object_path);

 public:
  /*!
   * \brief Create a LiteralDoc to represent None/null/empty value.
   * \param p The object path
   */
  static LiteralDoc None(const ffi::Optional<AccessPath>& p) {
    return LiteralDoc(ffi::Any(nullptr), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent integer.
   * \param v The integer value.
   * \param p The object path
   */
  static LiteralDoc Int(int64_t v, const ffi::Optional<AccessPath>& p) {
    return LiteralDoc(IntImm(DataType::Int(64), v), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent boolean.
   * \param v The boolean value.
   * \param p The object path
   */
  static LiteralDoc Boolean(bool v, const ffi::Optional<AccessPath>& p) {
    return LiteralDoc(IntImm(DataType::Bool(), v), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent float.
   * \param v The float value.
   * \param p The object path
   */
  static LiteralDoc Float(double v, const ffi::Optional<AccessPath>& p) {
    return LiteralDoc(FloatImm(DataType::Float(64), v), p);
  }
  /*!
   * \brief Create a LiteralDoc to represent string.
   * \param v The string value.
   * \param p The object path
   */
  static LiteralDoc Str(const ffi::String& v, const ffi::Optional<AccessPath>& p) {
    return LiteralDoc(v, p);
  }
  /*!
   * \brief Create a LiteralDoc to represent string.
   * \param v The string value.
   * \param p The object path
   */
  static LiteralDoc DataType(const runtime::DataType& v, const ffi::Optional<AccessPath>& p) {
    std::string dtype = v.is_void() ? "void" : runtime::DLDataTypeToString(v);
    return LiteralDoc::Str(dtype, p);
  }
  /*!
   * \brief Create a LiteralDoc to represent device.
   * \param v The device.
   * \param p The object path
   */
  static LiteralDoc Device(const DLDevice& v, const ffi::Optional<AccessPath>& p) {
    std::ostringstream os;
    runtime::operator<<(os, v);
    return LiteralDoc::Str(os.str(), p);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LiteralDoc, ExprDoc, LiteralDocNode);
};

/*!
 * \brief Doc that represents identifier.
 *
 * \sa IdDoc
 */
class IdDocNode : public ExprDocNode {
 public:
  /*! \brief The name of the identifier */
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IdDocNode>().def_ro("name", &IdDocNode::name);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IdDoc", IdDocNode, ExprDocNode);
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
  explicit IdDoc(ffi::String name);
  explicit IdDoc(std::nullptr_t) : ExprDoc(nullptr) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IdDoc, ExprDoc, IdDocNode);
};

/*!
 * \brief Doc that represents attribute access on another expression.
 *
 * \sa AttrAccessDoc
 */
class AttrAccessDocNode : public ExprDocNode {
 public:
  /*! \brief The target expression to be accessed */
  ExprDoc value{ffi::UnsafeInit()};
  /*! \brief The attribute to be accessed */
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AttrAccessDocNode>()
        .def_ro("value", &AttrAccessDocNode::value)
        .def_ro("name", &AttrAccessDocNode::name);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AttrAccessDoc", AttrAccessDocNode, ExprDocNode);
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
   * \param name The name of attribute to access.
   */
  explicit AttrAccessDoc(ExprDoc value, ffi::String name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AttrAccessDoc, ExprDoc, AttrAccessDocNode);
};

/*!
 * \brief Doc that represents index access on another expression.
 *
 * \sa IndexDoc
 */
class IndexDocNode : public ExprDocNode {
 public:
  /*! \brief The container value to be accessed */
  ExprDoc value{ffi::UnsafeInit()};
  /*!
   * \brief The indices to access
   *
   * Possible actual types:
   * - ExprDoc (single point access like a[1, 2])
   * - SliceDoc (slice access like a[1:5, 2])
   */
  ffi::Array<Doc> indices;  // Each element is union of: Slice / ExprDoc

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IndexDocNode>()
        .def_ro("value", &IndexDocNode::value)
        .def_ro("indices", &IndexDocNode::indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IndexDoc", IndexDocNode, ExprDocNode);
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
  explicit IndexDoc(ExprDoc value, ffi::Array<Doc> indices);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IndexDoc, ExprDoc, IndexDocNode);
};

/*!
 * \brief Doc that represents function call.
 *
 * \sa CallDoc
 */
class CallDocNode : public ExprDocNode {
 public:
  /*! \brief The callee of this function call */
  ExprDoc callee{ffi::UnsafeInit()};
  /*! \brief The positional arguments */
  ffi::Array<ExprDoc> args;
  /*! \brief The keys of keyword arguments */
  ffi::Array<ffi::String> kwargs_keys;
  /*!
   * \brief The values of keyword arguments.
   *
   * The i-th element is the value of the i-th key in `kwargs_keys`.
   * It must have the same length as `kwargs_keys`.
   */
  ffi::Array<ExprDoc> kwargs_values;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallDocNode>()
        .def_ro("callee", &CallDocNode::callee)
        .def_ro("args", &CallDocNode::args)
        .def_ro("kwargs_keys", &CallDocNode::kwargs_keys)
        .def_ro("kwargs_values", &CallDocNode::kwargs_values);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.CallDoc", CallDocNode, ExprDocNode);
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
  CallDoc(ExprDoc callee, ffi::Array<ExprDoc> args, ffi::Array<ffi::String> kwargs_keys,
          ffi::Array<ExprDoc> kwargs_values);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CallDoc, ExprDoc, CallDocNode);
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
    kUnaryStart = 0,
    kUSub = 1,    // -x
    kInvert = 2,  // ~x
    kNot = 3,     // not x
    kUnaryEnd = 4,

    // Binary operators
    kBinaryStart = 5,
    kAdd = 6,        // +
    kSub = 7,        // -
    kMult = 8,       // *
    kDiv = 9,        // /
    kFloorDiv = 10,  // // in Python
    kMod = 11,       // % in Python
    kPow = 12,       // ** in Python
    kLShift = 13,    // <<
    kRShift = 14,    // >>
    kBitAnd = 15,    // &
    kBitOr = 16,     // |
    kBitXor = 17,    // ^
    kLt = 18,        // <
    kLtE = 19,       // <=
    kEq = 20,        // ==
    kNotEq = 21,     // !=
    kGt = 22,        // >
    kGtE = 23,       // >=
    kAnd = 24,       // and
    kOr = 25,        // or
    kBinaryEnd = 26,

    // Special
    kSpecialStart = 27,
    kIfThenElse = 28,  // <operands[1]> if <operands[0]> else <operands[2]>
    kSpecialEnd = 29
  };

  /*! \brief The kind of operation (operator) */
  Kind kind;
  /*! \brief Operands of this expression */
  ffi::Array<ExprDoc> operands;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OperationDocNode>()
        .def_ro("kind", &OperationDocNode::kind)
        .def_ro("operands", &OperationDocNode::operands);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.OperationDoc", OperationDocNode, ExprDocNode);
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
  explicit OperationDoc(OperationDocNode::Kind kind, ffi::Array<ExprDoc> operands);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OperationDoc, ExprDoc, OperationDocNode);
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
  ffi::Array<IdDoc> args;
  /*! \brief The body of this anonymous function */
  ExprDoc body{ffi::UnsafeInit()};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LambdaDocNode>()
        .def_ro("args", &LambdaDocNode::args)
        .def_ro("body", &LambdaDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.LambdaDoc", LambdaDocNode, ExprDocNode);
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
  explicit LambdaDoc(ffi::Array<IdDoc> args, ExprDoc body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LambdaDoc, ExprDoc, LambdaDocNode);
};

/*!
 * \brief Doc that represents tuple literal.
 *
 * \sa TupleDoc
 */
class TupleDocNode : public ExprDocNode {
 public:
  /*! \brief Elements of tuple */
  ffi::Array<ExprDoc> elements;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleDocNode>().def_ro("elements", &TupleDocNode::elements);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.TupleDoc", TupleDocNode, ExprDocNode);
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
  TupleDoc() : ExprDoc(ffi::make_object<TupleDocNode>()) {}
  /*!
   * \brief Constructor of TupleDoc
   * \param elements Elements of tuple.
   */
  explicit TupleDoc(ffi::Array<ExprDoc> elements);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleDoc, ExprDoc, TupleDocNode);
};

/*!
 * \brief Doc that represents list literal.
 *
 * \sa AttrAccessDoc
 */
class ListDocNode : public ExprDocNode {
 public:
  /*! \brief Elements of list */
  ffi::Array<ExprDoc> elements;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ListDocNode>().def_ro("elements", &ListDocNode::elements);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ListDoc", ListDocNode, ExprDocNode);
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
  ListDoc() : ExprDoc(ffi::make_object<ListDocNode>()) {}
  /*!
   * \brief Constructor of ListDoc
   * \param elements Elements of list.
   */
  explicit ListDoc(ffi::Array<ExprDoc> elements);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ListDoc, ExprDoc, ListDocNode);
};

/*!
 * \brief Doc that represents dictionary literal.
 *
 * \sa AttrAccessDoc
 */
class DictDocNode : public ExprDocNode {
 public:
  /*! \brief keys of dictionary */
  ffi::Array<ExprDoc> keys;
  /*!
   * \brief Values of dictionary
   *
   * The i-th element is the value of the i-th element of `keys`.
   * It must have the same length as `keys`.
   */
  ffi::Array<ExprDoc> values;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DictDocNode>()
        .def_ro("keys", &DictDocNode::keys)
        .def_ro("values", &DictDocNode::values);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.DictDoc", DictDocNode, ExprDocNode);
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
  DictDoc() : ExprDoc(ffi::make_object<DictDocNode>()) {}
  /*!
   * \brief Constructor of DictDoc
   * \param keys Keys of dictionary.
   * \param values Values of dictionary, must have same length as `keys`.
   */
  explicit DictDoc(ffi::Array<ExprDoc> keys, ffi::Array<ExprDoc> values);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DictDoc, ExprDoc, DictDocNode);
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
  ffi::Optional<ExprDoc> start;
  /*! \brief The exclusive end of slice */
  ffi::Optional<ExprDoc> stop;
  /*! \brief The step of slice */
  ffi::Optional<ExprDoc> step;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SliceDocNode>()
        .def_ro("start", &SliceDocNode::start)
        .def_ro("stop", &SliceDocNode::stop)
        .def_ro("step", &SliceDocNode::step);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.SliceDoc", SliceDocNode, DocNode);
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
   * \param stop The exclusive end of slice.
   * \param step The step of slice.
   */
  explicit SliceDoc(ffi::Optional<ExprDoc> start, ffi::Optional<ExprDoc> stop,
                    ffi::Optional<ExprDoc> step);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SliceDoc, Doc, SliceDocNode);
};

/*!
 * \brief Doc that represents assign statement.
 *
 * \sa AssignDoc
 */
class AssignDocNode : public StmtDocNode {
 public:
  /*! \brief The left hand side of the assignment */
  ExprDoc lhs{ffi::UnsafeInit()};
  /*!
   * \brief The right hand side of the assignment.
   *
   * If null, this doc represents declaration, e.g. `A: T.Buffer((1,2))`
   * */
  ffi::Optional<ExprDoc> rhs;
  /*! \brief The type annotation of this assignment. */
  ffi::Optional<ExprDoc> annotation;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AssignDocNode>()
        .def_ro("lhs", &AssignDocNode::lhs)
        .def_ro("rhs", &AssignDocNode::rhs)
        .def_ro("annotation", &AssignDocNode::annotation);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AssignDoc", AssignDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of AssignDocNode.
 *
 * \sa AssignDoc
 */
class AssignDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of AssignDoc.
   * \param lhs The left hand side of the assignment.
   * \param rhs The right hand side of the assignment.
   * \param annotation The type annotation of this assignment.
   */
  explicit AssignDoc(ExprDoc lhs, ffi::Optional<ExprDoc> rhs, ffi::Optional<ExprDoc> annotation);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssignDoc, StmtDoc, AssignDocNode);
};

/*!
 * \brief Doc that represent if-then-else statement.
 *
 * \sa IfDoc
 */
class IfDocNode : public StmtDocNode {
 public:
  /*! \brief The predicate of the if-then-else statement. */
  ExprDoc predicate{ffi::UnsafeInit()};
  /*! \brief The then branch of the if-then-else statement. */
  ffi::Array<StmtDoc> then_branch;
  /*! \brief The else branch of the if-then-else statement. */
  ffi::Array<StmtDoc> else_branch;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfDocNode>()
        .def_ro("predicate", &IfDocNode::predicate)
        .def_ro("then_branch", &IfDocNode::then_branch)
        .def_ro("else_branch", &IfDocNode::else_branch);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.IfDoc", IfDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of IfDocNode.
 *
 * \sa IfDocNode
 */
class IfDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of IfDoc.
   * \param predicate The predicate of the if-then-else statement.
   * \param then_branch The then branch of the if-then-else statement.
   * \param else_branch The else branch of the if-then-else statement.
   */
  explicit IfDoc(ExprDoc predicate, ffi::Array<StmtDoc> then_branch,
                 ffi::Array<StmtDoc> else_branch);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfDoc, StmtDoc, IfDocNode);
};

/*!
 * \brief Doc that represents while statement.
 *
 * \sa WhileDoc
 */
class WhileDocNode : public StmtDocNode {
 public:
  /*! \brief The predicate of the while statement. */
  ExprDoc predicate{ffi::UnsafeInit()};
  /*! \brief The body of the while statement. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WhileDocNode>()
        .def_ro("predicate", &WhileDocNode::predicate)
        .def_ro("body", &WhileDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.WhileDoc", WhileDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of WhileDocNode.
 *
 * \sa WhileDocNode
 */
class WhileDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of WhileDoc.
   * \param predicate The predicate of the while statement.
   * \param body The body of the while statement.
   */
  explicit WhileDoc(ExprDoc predicate, ffi::Array<StmtDoc> body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WhileDoc, StmtDoc, WhileDocNode);
};

/*!
 * \brief Doc that represents for statement.
 *
 * Example:
 * for 'lhs' in 'rhs':
 *   'body...'
 *
 * \sa ForDoc
 */
class ForDocNode : public StmtDocNode {
 public:
  /*! \brief The left hand side of the assignment of iterating variable. */
  ExprDoc lhs{ffi::UnsafeInit()};
  /*! \brief The right hand side of the assignment of iterating variable. */
  ExprDoc rhs{ffi::UnsafeInit()};
  /*! \brief The body of the for statement. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ForDocNode>()
        .def_ro("lhs", &ForDocNode::lhs)
        .def_ro("rhs", &ForDocNode::rhs)
        .def_ro("body", &ForDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ForDoc", ForDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ForDocNode.
 *
 * \sa ForDocNode
 */
class ForDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ForDoc.
   * \param lhs The left hand side of the assignment of iterating variable.
   * \param rhs The right hand side of the assignment of iterating variable.
   * \param body The body of the for statement.
   */
  explicit ForDoc(ExprDoc lhs, ExprDoc rhs, ffi::Array<StmtDoc> body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ForDoc, StmtDoc, ForDocNode);
};

/*!
 * \brief Doc that represents special scopes.
 *
 * Specifically, this means the with statement in Python:
 *
 * with 'rhs' as 'lhs':
 *   'body...'
 *
 * \sa ScopeDoc
 */
class ScopeDocNode : public StmtDocNode {
 public:
  /*! \brief The name of the scoped variable. */
  ffi::Optional<ExprDoc> lhs{std::nullopt};
  /*! \brief The value of the scoped variable. */
  ExprDoc rhs{ffi::UnsafeInit()};
  /*! \brief The body of the scope doc. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScopeDocNode>()
        .def_ro("lhs", &ScopeDocNode::lhs)
        .def_ro("rhs", &ScopeDocNode::rhs)
        .def_ro("body", &ScopeDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ScopeDoc", ScopeDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ScopeDocNode.
 *
 * \sa ScopeDocNode
 */
class ScopeDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ScopeDoc.
   * \param lhs The name of the scoped variable.
   * \param rhs The value of the scoped variable.
   * \param body The body of the scope doc.
   */
  explicit ScopeDoc(ffi::Optional<ExprDoc> lhs, ExprDoc rhs, ffi::Array<StmtDoc> body);

  /*!
   * \brief Constructor of ScopeDoc.
   * \param rhs The value of the scoped variable.
   * \param body The body of the scope doc.
   */
  explicit ScopeDoc(ExprDoc rhs, ffi::Array<StmtDoc> body);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScopeDoc, StmtDoc, ScopeDocNode);
};

/*!
 * \brief Doc that represents an expression as statement.
 *
 * \sa ExprStmtDoc
 */
class ExprStmtDocNode : public StmtDocNode {
 public:
  /*! \brief The expression represented by this doc. */
  ExprDoc expr{ffi::UnsafeInit()};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExprStmtDocNode>().def_ro("expr", &ExprStmtDocNode::expr);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ExprStmtDoc", ExprStmtDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ExprStmtDocNode.
 *
 * \sa ExprStmtDocNode
 */
class ExprStmtDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ExprStmtDoc.
   * \param expr The expression represented by this doc.
   */
  explicit ExprStmtDoc(ExprDoc expr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExprStmtDoc, StmtDoc, ExprStmtDocNode);
};

/*!
 * \brief Doc that represents assert statement.
 *
 * \sa AssertDoc
 */
class AssertDocNode : public StmtDocNode {
 public:
  /*! \brief The expression to test. */
  ExprDoc test{ffi::UnsafeInit()};
  /*! \brief The optional error message when assertion failed. */
  ffi::Optional<ExprDoc> msg{std::nullopt};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AssertDocNode>()
        .def_ro("test", &AssertDocNode::test)
        .def_ro("msg", &AssertDocNode::msg);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.AssertDoc", AssertDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of AssertDocNode.
 *
 * \sa AssertDocNode
 */
class AssertDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of AssertDoc.
   * \param test The expression to test.
   * \param msg The optional error message when assertion failed.
   */
  explicit AssertDoc(ExprDoc test, ffi::Optional<ExprDoc> msg = std::nullopt);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertDoc, StmtDoc, AssertDocNode);
};

/*!
 * \brief Doc that represents return statement.
 *
 * \sa ReturnDoc
 */
class ReturnDocNode : public StmtDocNode {
 public:
  /*! \brief The value to return. */
  ExprDoc value{ffi::UnsafeInit()};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReturnDocNode>().def_ro("value", &ReturnDocNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ReturnDoc", ReturnDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ReturnDocNode.
 *
 * \sa ReturnDocNode
 */
class ReturnDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ReturnDoc.
   * \param value The value to return.
   */
  explicit ReturnDoc(ExprDoc value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ReturnDoc, StmtDoc, ReturnDocNode);
};

/*!
 * \brief Doc that represents function definition.
 *
 * \sa FunctionDoc
 */
class FunctionDocNode : public StmtDocNode {
 public:
  /*! \brief The name of function. */
  IdDoc name{ffi::UnsafeInit{}};
  /*!
   * \brief The arguments of function.
   *
   * The `lhs` means argument name,
   * `annotation` means argument type,
   * and `rhs` means default value.
   */
  ffi::Array<AssignDoc> args;
  /*! \brief Decorators of function. */
  ffi::Array<ExprDoc> decorators;
  /*! \brief The return type of function. */
  ffi::Optional<ExprDoc> return_type{std::nullopt};
  /*! \brief The body of function. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionDocNode>()
        .def_ro("name", &FunctionDocNode::name)
        .def_ro("args", &FunctionDocNode::args)
        .def_ro("decorators", &FunctionDocNode::decorators)
        .def_ro("return_type", &FunctionDocNode::return_type)
        .def_ro("body", &FunctionDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.FunctionDoc", FunctionDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of FunctionDocNode.
 *
 * \sa FunctionDocNode
 */
class FunctionDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of FunctionDoc.
   * \param name The name of function..
   * \param args The arguments of function.
   * \param decorators The decorator of function.
   * \param return_type The return type of function.
   * \param body The body of function.
   */
  explicit FunctionDoc(IdDoc name, ffi::Array<AssignDoc> args, ffi::Array<ExprDoc> decorators,
                       ffi::Optional<ExprDoc> return_type, ffi::Array<StmtDoc> body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionDoc, StmtDoc, FunctionDocNode);
};

/*!
 * \brief Doc that represents class definition.
 *
 * \sa ClassDoc
 */
class ClassDocNode : public StmtDocNode {
 public:
  /*! \brief The name of class. */
  IdDoc name{ffi::UnsafeInit{}};
  /*! \brief Decorators of class. */
  ffi::Array<ExprDoc> decorators;
  /*! \brief The body of class. */
  ffi::Array<StmtDoc> body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ClassDocNode>()
        .def_ro("name", &ClassDocNode::name)
        .def_ro("decorators", &ClassDocNode::decorators)
        .def_ro("body", &ClassDocNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.ClassDoc", ClassDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of ClassDocNode.
 *
 * \sa ClassDocNode
 */
class ClassDoc : public StmtDoc {
 public:
  /*!
   * \brief Constructor of ClassDoc.
   * \param name The name of class.
   * \param decorators The decorator of class.
   * \param body The body of class.
   */
  explicit ClassDoc(IdDoc name, ffi::Array<ExprDoc> decorators, ffi::Array<StmtDoc> body);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ClassDoc, StmtDoc, ClassDocNode);
};

/*!
 * \brief Doc that represents comment.
 *
 * \sa CommentDoc
 */
class CommentDocNode : public StmtDocNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CommentDocNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.CommentDoc", CommentDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of CommentDocNode.
 *
 * \sa CommentDocNode
 */
class CommentDoc : public StmtDoc {
 public:
  explicit CommentDoc(ffi::String comment);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CommentDoc, StmtDoc, CommentDocNode);
};

/*!
 * \brief Doc that represents docstring.
 *
 * \sa DocStringDoc
 */
class DocStringDocNode : public StmtDocNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DocStringDocNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.printer.DocStringDoc", DocStringDocNode, StmtDocNode);
};

/*!
 * \brief Reference type of DocStringDocNode.
 *
 * \sa DocStringDocNode
 */
class DocStringDoc : public StmtDoc {
 public:
  explicit DocStringDoc(ffi::String docs);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DocStringDoc, StmtDoc, DocStringDocNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_H_
