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
 * \file printer/tir_text_printer.cc
 * \brief Printer to print out the IR text format
 *        that can be parsed by a parser.
 */

#include <tvm/ir/module.h>
#include <tvm/ir/type.h>
#include <tvm/ir/type_functor.h>
#include <tvm/node/serialization.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <string>

#include "doc.h"
#include "meta_data.h"

namespace tvm {
namespace tir {

/*!
 *  \brief Meta node collector
 *  If we decide to put some node into meta, then all the sub-nodes inside
 *  it need to be put in meta as well, since when parsing we need to know
 *  whether two refs are the same
 */
class MetaCollector : public StmtExprVisitor {
 public:
  explicit MetaCollector(TextMetaDataContext* meta) : meta_(meta) {}

  void Collect(const ObjectRef& n) {
    // these nodes can be print directly(StringLiteral or use identifier to identify)
    if (!n.defined() || n.as<StringImmNode>() || n.as<StringObj>() || n.as<SizeVarNode>()
        || n.as<VarNode>() || n.as<BufferNode>() || n.as<IterVarNode>()) {
      return;
    }
    if (n->IsInstance<StmtNode>()) {
      VisitStmt(Downcast<Stmt>(n));
    } else if (n->IsInstance<PrimExprNode>()) {
      VisitExpr(Downcast<PrimExpr>(n));
    }
  }

  void VisitStmt(const Stmt& n) override {
    meta_->GetMetaNode(n);
    StmtVisitor::VisitStmt(n);
  }

  void VisitExpr(const PrimExpr& n) override {
    meta_->GetMetaNode(n);
    ExprVisitor::VisitExpr(n);
  }

 private:
  TextMetaDataContext* meta_;
};

class TIRTextPrinter : public StmtFunctor<Doc(const Stmt&)>,
                       public ExprFunctor<Doc(const PrimExpr&)>,
                       public TypeFunctor<Doc(const Type&)> {
 public:
  explicit TIRTextPrinter(bool show_meta) : show_meta_(show_meta), meta_collector_(&meta_) {}

  /*! \brief Print the node */
  Doc Print(const ObjectRef& node);

 private:
  /*! \brief whether show meta data */
  bool show_meta_;
  /*! \brief meta data context */
  TextMetaDataContext meta_;
  /*! \brief meta collector */
  MetaCollector meta_collector_;
  /*! \brief Map from Var to Doc */
  std::unordered_map<Var, Doc, ObjectHash, ObjectEqual> memo_var_;
  /*! \brief Map from Buffer to Doc */
  std::unordered_map<Buffer, Doc, ObjectHash, ObjectEqual> memo_buf_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;

  Doc VisitExpr_(const IntImmNode* op) override;
  Doc VisitExpr_(const FloatImmNode* op) override;
  Doc VisitExpr_(const StringImmNode* op) override;
  Doc VisitExpr_(const CastNode* op) override;
  Doc VisitExpr_(const VarNode* op) override;
  Doc VisitExpr_(const AddNode* op) override;
  Doc VisitExpr_(const SubNode* op) override;
  Doc VisitExpr_(const MulNode* op) override;
  Doc VisitExpr_(const DivNode* op) override;
  Doc VisitExpr_(const ModNode* op) override;
  Doc VisitExpr_(const FloorDivNode* op) override;
  Doc VisitExpr_(const FloorModNode* op) override;
  Doc VisitExpr_(const MinNode* op) override;
  Doc VisitExpr_(const MaxNode* op) override;
  Doc VisitExpr_(const EQNode* op) override;
  Doc VisitExpr_(const NENode* op) override;
  Doc VisitExpr_(const LTNode* op) override;
  Doc VisitExpr_(const LENode* op) override;
  Doc VisitExpr_(const GTNode* op) override;
  Doc VisitExpr_(const GENode* op) override;
  Doc VisitExpr_(const AndNode* op) override;
  Doc VisitExpr_(const OrNode* op) override;
  Doc VisitExpr_(const NotNode* op) override;
  Doc VisitExpr_(const SelectNode* op) override;
  Doc VisitExpr_(const BufferLoadNode* op) override;
  Doc VisitExpr_(const LoadNode* op) override;
  Doc VisitExpr_(const RampNode* op) override;
  Doc VisitExpr_(const BroadcastNode* op) override;
  Doc VisitExpr_(const LetNode* op) override;
  Doc VisitExpr_(const CallNode* op) override;
  Doc VisitExpr_(const ShuffleNode* op) override;
  Doc VisitExpr_(const ReduceNode* op) override;
  Doc VisitExprDefault_(const Object* op) override;

  Doc VisitStmt_(const LetStmtNode* op) override;
  Doc VisitStmt_(const AttrStmtNode* op) override;
  Doc VisitStmt_(const AssertStmtNode* op) override;
  Doc VisitStmt_(const StoreNode* op) override;
  Doc VisitStmt_(const BufferStoreNode* op) override;
  Doc VisitStmt_(const BufferRealizeNode* op) override;
  Doc VisitStmt_(const AllocateNode* op) override;
  Doc VisitStmt_(const FreeNode* op) override;
  Doc VisitStmt_(const IfThenElseNode* op) override;
  Doc VisitStmt_(const SeqStmtNode* op) override;
  Doc VisitStmt_(const EvaluateNode* op) override;
  Doc VisitStmt_(const ForNode* op) override;
  Doc VisitStmt_(const PrefetchNode* op) override;
  Doc VisitStmtDefault_(const Object* op) override;

  Doc VisitType_(const PrimTypeNode* node) override;
  Doc VisitType_(const PointerTypeNode* node) override;
  Doc VisitType_(const TupleTypeNode* node) override;

  Doc PrintIRModule(const IRModule& module);
  Doc PrintPrimFunc(const PrimFunc& primFunc);
  Doc PrintArray(const ArrayNode* op);
  Doc PrintIterVar(const IterVarNode* op);
  Doc PrintRange(const RangeNode* op);
  Doc PrintBuffer(const BufferNode* op);
  Doc PrintString(const StringObj* op) {
    return Doc::StrLiteral(op->data);
  }

  /*!
   * \brief special method to print out data type
   * \param dtype The data type
   */
  static Doc PrintDType(DataType dtype) {
    return Doc::Text(runtime::DLDataType2String(dtype));
  }

  /*!
   * \brief special method to print out const scalar
   * \param dtype The data type
   * \param data The pointer to hold the data.
   */
  template <typename T>
  static Doc PrintConstScalar(DataType dtype, const T& data) {
    Doc doc;
    std::ostringstream os;
    os << data;
    if (dtype == DataType::Int(32)) {
      doc << Doc::Text(os.str());
    } else {
      doc << PrintDType(dtype) << "(" << Doc::Text(os.str()) << ")";
    }
    return doc;
  }

  Doc GetUniqueName(std::string prefix) {
    // std::replace(prefix.begin(), prefix.end(), '.', '_');
    std::string unique_prefix = prefix;
    auto it = name_alloc_map_.find(prefix);
    if (it != name_alloc_map_.end()) {
      while (name_alloc_map_.count(
          unique_prefix = prefix + "_" + std::to_string(++it->second)) > 0) {}
    }
    name_alloc_map_[unique_prefix] = 0;
    return Doc::Text(unique_prefix);
  }

  Doc AllocVar(const Var& var) {
    const auto& it = memo_var_.find(var);
    if (it != memo_var_.end()) {
      return it->second;
    }
    std::string name = var->name_hint;
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "v" + name;
    }
    Doc val = GetUniqueName(name);
    memo_var_[var] = val;
    return val << ": " << Print(GetType(var));
  }

  Doc AllocBuf(const Buffer& buffer) {
    const auto& it = memo_buf_.find(buffer);
    if (it != memo_buf_.end()) {
      return it->second;
    }
    std::string name = buffer->name;
    if (name.length() == 0 || !std::isalpha(name[0])) {
      name = "buf_" + name;
    }
    Doc val = GetUniqueName(name);
    memo_buf_[buffer] = val;
    return val;
  }

  /*!
   * \brief special method to render vectors of docs with a separator
   * \param vec vector of docs
   * \param sep separator
   */
  static Doc PrintSep(const std::vector<Doc>& vec, const Doc& sep) {
    Doc seq;
    if (vec.size() != 0) {
      seq = vec[0];
      for (size_t i = 1; i < vec.size(); i++) {
        seq << sep << vec[i];
      }
    }
    return seq;
  }

  /*!
   * \brief dump meta info
   * \return Doc with meta info
   */
  Doc DumpMeta() {
    if (show_meta_) {
      return Doc::Text("__tvm_meta__ = ")
          << (meta_.empty() ? Doc::Text("None") : meta_.GetMetaSection());
    } else {
      return Doc::Text("");
    }
  }

  Doc PrintBody(const Stmt& body, bool indent = true) {
    Doc doc;
    if (body->IsInstance<SeqStmtNode>()) return Print(body);
    doc << " {" << Doc::Indent(2, Doc::NewLine() << Print(body)) << Doc::NewLine() << "}";
    return doc;
  }
};

Doc TIRTextPrinter::Print(const ObjectRef& node) {
  if (!node.defined()) return Doc::Text("(nullptr)");
  if (node->IsInstance<StmtNode>()) {
    return VisitStmt(Downcast<Stmt>(node));
  } else if (node->IsInstance<PrimExprNode>()) {
    return VisitExpr(Downcast<PrimExpr>(node));
  } else if (node->IsInstance<TypeNode>()) {
    return VisitType(Downcast<Type>(node));
  } else if (node->IsInstance<PrimFuncNode>()) {
    return PrintPrimFunc(Downcast<PrimFunc>(node));
  } else if (node->IsInstance<IRModuleNode>()) {
    return PrintIRModule(Downcast<IRModule>(node));
  } else if (node->IsInstance<ArrayNode>()) {
    return PrintArray(node.as<ArrayNode>());
  } else if (node->IsInstance<IterVarNode>()) {
    return PrintIterVar(node.as<IterVarNode>());
  } else if (node->IsInstance<RangeNode>()) {
    return PrintRange(node.as<RangeNode>());
  } else if (node->IsInstance<BufferNode>()) {
    return PrintBuffer(node.as<BufferNode>());
  } else if (node->IsInstance<StringObj>()) {
    return PrintString(node.as<StringObj>());
  } else {
    return this->meta_.GetMetaNode(node);
  }
}

Doc TIRTextPrinter::PrintPrimFunc(const PrimFunc& primFunc) {
  const auto* op = primFunc.operator->();
  const auto& signature = op->func_type_annotation();
  // collect Meta in DictAttr
  for (const auto& it : primFunc->attrs->dict) {
    meta_collector_.Collect(it.second);
  }
  // collect buffers in buffer_map
  memo_var_.clear();
  memo_buf_.clear();
  for (const auto& it : op->buffer_map) {
    memo_buf_[it.second] = AllocBuf(it.second);
  }
  // print PrimFunc
  Doc doc;
  doc << "primfn" << "(";
  // print params and its type annotation
  std::vector<Doc> params;
  for (const auto& param : op->params) {
    params.push_back(Print(param));
  }
  Doc sep;
  doc << PrintSep(params, Doc::Indent(9, Doc::Text(", "))) << ")";
  // print return type
  doc << " -> " << Print(signature->ret_type);
  // print attr
  Doc attr_doc;
  std::vector<Doc> attr_docs;
  for (const auto& it : op->attrs->dict) {
    attr_docs.push_back(Doc::StrLiteral(it.first) << ": " << Print(it.second));
  }
  attr_doc << Doc::NewLine() << "attr = {" << PrintSep(attr_docs, Doc::Text(", ")) << "}";
  doc << Doc::Indent(2, attr_doc);
  // print all the buffers in the tree
  Doc buffer_doc;
  std::vector<Doc> buffer_docs;
  for (const auto& it : memo_buf_) {
    const auto& buf = it.first;
    buffer_docs.push_back(Print(buf)
                          << Doc::Text(": Buffer(") << Print(buf->data) << ", "
                          << PrintDType(buf->dtype) << ", " << Print(buf->shape) << ", "
                          << Print(buf->strides));
    if (!is_zero(buf->elem_offset)) {
      buffer_docs.back() << ", elem_offset=" << Print(buf->elem_offset);
    }
    if (buf->scope != "global") {
      buffer_docs.back() << ", scope=" << Doc::StrLiteral(buf->scope);
    }
    if (buf->data_alignment != 128) {
      buffer_docs.back() << ", align=" << buf->data_alignment;
    }
    if (buf->offset_factor != 1) {
      buffer_docs.back() << ", offset_factor=" << buf->offset_factor;
    }
    if (buf->buffer_type != 1) {
      buffer_docs.back() << ", type=" << Doc::StrLiteral("auto");
    }
    buffer_docs.back() << ")";
  }
  buffer_doc << Doc::NewLine() << "buffers = {";
  buffer_doc << PrintSep(buffer_docs, Doc::Indent(9, Doc::Text(",") << Doc::NewLine()));
  doc << Doc::Indent(2, buffer_doc) << "}";
  // print buffer_map
  std::vector<Doc> buffer_map_doc;
  for (const auto& it : op->buffer_map) {
    buffer_map_doc.push_back(Print(it.first) << ": " << Print(it.second));
  }
  doc << Doc::Indent(2, Doc::NewLine()
      << "buffer_map = {" << PrintSep(buffer_map_doc, Doc::Text(", ")) << "}");
  doc << PrintBody(op->body);
  return doc;
}

Doc TIRTextPrinter::PrintIRModule(const IRModule& module) {
  const auto* op = module.operator->();
  Doc doc;

  Doc body;
  body << Doc::NewLine();
  std::vector<Doc> functions;
  for (auto it = op->functions.begin(); it != op->functions.end(); ++it) {
    if ((*it).second.as<PrimFuncNode>()) {
      functions.push_back(Print((*it).second));
    }
  }
  body << TIRTextPrinter::PrintSep(functions, Doc::NewLine() << Doc::NewLine());
  body << Doc::NewLine() << DumpMeta();
  doc << Doc::Indent(0, body);
  return doc;
}

Doc TIRTextPrinter::PrintArray(const ArrayNode* op) {
  Doc doc;
  doc << '[';
  for (size_t i = 0; i < op->data.size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    doc << Print(op->data[i]);
  }
  doc << ']';
  return doc;
}

Doc TIRTextPrinter::PrintIterVar(const IterVarNode* op) {
  Doc doc;
  doc << "IterVar(" << Print(op->var);
  if (op->dom.defined()) {
    doc << ", [" << Print(op->dom) << "], ";
  } else {
    doc << ", " << Print(op->dom) << ", ";
  }
  doc << Doc::StrLiteral(IterVarType2String(op->iter_type)) << ", ";
  doc << Doc::StrLiteral(op->thread_tag) << ")";
  return doc;
}

Doc TIRTextPrinter::PrintRange(const RangeNode* op) {
  return Print(op->min) << ":" << Print(op->min + op->extent);
}

Doc TIRTextPrinter::PrintBuffer(const BufferNode* op) {
  const Buffer& buffer = GetRef<Buffer>(op);
  CHECK_GT(memo_buf_.count(buffer), 0);
  return meta_.InMeta(buffer) ? meta_.GetMetaNode(buffer) : memo_buf_[buffer];
}

Doc TIRTextPrinter::VisitExprDefault_(const Object* op) {
  return this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

Doc TIRTextPrinter::VisitStmtDefault_(const Object* op) {
  return this->meta_.GetMetaNode(GetRef<ObjectRef>(op));
}

Doc TIRTextPrinter::VisitExpr_(const IntImmNode* op) {
  return PrintConstScalar<int64_t>(op->dtype, op->value);
}

Doc TIRTextPrinter::VisitExpr_(const FloatImmNode* op) {
  return PrintConstScalar<double>(op->dtype, op->value);
}

Doc TIRTextPrinter::VisitExpr_(const StringImmNode* op) { return Doc::StrLiteral(op->value); }

Doc TIRTextPrinter::VisitExpr_(const CastNode* op) {
  Doc doc;
  doc << "cast(" << PrintDType(op->dtype) << ", " << Print(op->value) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const VarNode* op) {
  const Var& var = GetRef<Var>(op);
  return meta_.InMeta(var) ? meta_.GetMetaNode(var) : AllocVar(GetRef<Var>(op));
}

#define TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(OpName, OpString)     \
  Doc TIRTextPrinter::VisitExpr_(const OpName* op) {               \
    Doc doc;                                                       \
    doc << '(' << Print(op->a) << OpString << Print(op->b) << ")"; \
    return doc;                                                    \
  }

TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(AddNode, " + ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(SubNode, " - ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(MulNode, "*")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(DivNode, " / ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(ModNode, " % ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(EQNode, " == ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(NENode, " != ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(LTNode, " < ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(LENode, " <= ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(GTNode, " > ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(GENode, " >= ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(AndNode, " and ")
TVM_DECLARE_TIR_HYBRID_PRINTER_BINOP(OrNode, " or ")

Doc TIRTextPrinter::VisitExpr_(const FloorDivNode* op) {
  Doc doc;
  doc << "floordiv(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const FloorModNode* op) {
  Doc doc;
  doc << "floormod(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const MinNode* op) {
  Doc doc;
  doc << "min(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const MaxNode* op) {
  Doc doc;
  doc << "max(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const NotNode* op) {
  Doc doc;
  doc << "!" << Print(op->a);
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const SelectNode* op) {
  Doc doc;
  doc << "select(" << Print(op->condition) << ", " << Print(op->true_value) << ", "
      << Print(op->false_value);
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const BufferLoadNode* op) {
  Doc doc;
  doc << Print(op->buffer) << Print(op->indices);
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const LoadNode* op) {
  Doc doc;
  doc << "load(" << PrintDType(op->dtype) << ", "
      << Print(op->buffer_var) << "[" << Print(op->index) << "])";
  if (!is_one(op->predicate)) {
    doc << " if " << Print(op->predicate);
  }
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const RampNode* op) {
  Doc doc;
  doc << "ramp(" << Print(op->base) << ", " << Print(op->stride) << ", " << op->lanes << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const BroadcastNode* op) {
  Doc doc;
  doc << "broadcast(" << Print(op->value) << ", " << op->lanes << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const LetNode* op) {
  Doc doc;
  doc << "let " << Print(op->var) << " = " << Print(op->value) << " in " << Print(op->body);
  return doc;
}

inline const char* CallType2String(CallNode::CallType t) {
  switch (t) {
    case CallNode::Extern:return "extern";
    case CallNode::ExternCPlusPlus:return "extern_cpp";
    case CallNode::PureExtern:return "pure_extern";
    case CallNode::Halide:return "halide";
    case CallNode::Intrinsic:return "intrin";
    case CallNode::PureIntrinsic:return "pure_intrin";
  }
  LOG(FATAL) << "Unknown CallType";
  return "Unknown";
}

Doc TIRTextPrinter::VisitExpr_(const CallNode* op) {
  Doc doc;
  doc << "call(" << Doc::StrLiteral(op->name) << ", " << Print(op->args)
      << ", " << PrintDType(op->dtype) << ", "
      << Doc::StrLiteral(CallType2String(op->call_type)) << ", "
      << op->value_index << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const ShuffleNode* op) {
  Doc doc;
  doc << "shuffle(" << Print(op->vectors) << ", " << Print(op->indices) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const ReduceNode* op) {
  Doc doc;
  doc << "reduce(" << Print(op->combiner) << ", " << Print(op->source) << ", " << Print(op->axis)
      << ", " << op->value_index << ")";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const LetStmtNode* op) {
  Doc doc;
  doc << "let " << Print(op->var) << " = " << Print(op->value) << PrintBody(op->body);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const AttrStmtNode* op) {
  Doc doc;
  meta_collector_.Collect(op->node);
  doc << "attr [" << Print(op->node) << "] " << Doc::StrLiteral(op->attr_key) << " = "
      << Print(op->value);
  if (op->body->IsInstance<SeqStmtNode>()) {
    doc << PrintBody(op->body);
  } else {
    doc << ";" << Doc::NewLine() << Print(op->body);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const AssertStmtNode* op) {
  Doc doc;
  doc << "assert(" << Print(op->condition) << ", " << Print(op->message) << ")"
      << PrintBody(op->body);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const StoreNode* op) {
  Doc doc;
  doc << Print(op->buffer_var) << "[" << Print(op->index) << "] = " << Print(op->value);
  if (!is_one(op->predicate)) {
    doc << " if " << Print(op->predicate);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const BufferStoreNode* op) {
  Doc doc;
  doc << Print(op->buffer) << Print(op->indices) << " = " << Print(op->value);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const BufferRealizeNode* op) {
  Doc doc;
  doc << "realize(" << Print(op->buffer) << ", " << Print(op->bounds) << ", "
      << Print(op->condition) << PrintBody(op->body) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const AllocateNode* op) {
  Doc doc;
  doc << "allocate(" << Print(op->buffer_var) << ", " << PrintDType(op->dtype) << ", "
      << Print(op->extents) << ")";
  if (!is_one(op->condition)) {
    doc << " if " << Print(op->condition);
  }
  if (op->body->IsInstance<SeqStmtNode>()) {
    doc << PrintBody(op->body);
  } else {
    doc << ";" << Doc::NewLine() << Print(op->body);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const FreeNode* op) {
  Doc doc;
  doc << "free(" << Print(op->buffer_var) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const IfThenElseNode* op) {
  Doc doc;
  doc << "if " << Print(op->condition) << PrintBody(op->then_case);
  if (!is_one(op->condition) && op->else_case.defined()) {
    doc << " else" << PrintBody(op->else_case);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const SeqStmtNode* op) {
  std::vector<Doc> stmts;
  Doc seq_doc, doc;
  for (Stmt stmt : op->seq) {
    seq_doc << Doc::NewLine() << Print(stmt);
  }
  doc << " {" << Doc::Indent(2, seq_doc) << Doc::NewLine() << "}";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const EvaluateNode* op) {
  Doc doc;
  doc << Print(op->value);
  return doc;
}

inline const char* ForType2String(ForType t) {
  switch (t) {
    case ForType::Serial:return "serial";
    case ForType::Parallel:return "parallel";
    case ForType::Vectorized:return "vectorized";
    case ForType::Unrolled:return "unroll";
  }
  LOG(FATAL) << "Unknown ForType";
  return "Unknown";
}

Doc TIRTextPrinter::VisitStmt_(const ForNode* op) {
  Doc doc;
  doc << "for (" << Print(op->loop_var) << ", " << Print(op->min) << ", "
      << Print(op->min + op->extent) << ")";
  if (op->for_type != ForType::Serial) {
    doc << " " << Doc::StrLiteral(ForType2String(op->for_type));
  }
  doc << PrintBody(op->body);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const PrefetchNode* op) {
  Doc doc;
  doc << "prefetch(" << Print(op->buffer) << ", " << Print(op->bounds) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitType_(const PrimTypeNode* node) {
  Doc doc;
  doc << PrintDType(node->dtype);
  return doc;
}

Doc TIRTextPrinter::VisitType_(const PointerTypeNode* node) {
  Doc doc;
  doc << "Pointer(" << Print(node->element_type) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitType_(const TupleTypeNode* node) {
  std::vector<Doc> fields;
  for (Type field : node->fields) {
    fields.push_back(Print(field));
  }
  Doc doc;
  doc << "(" << Doc::Concat(fields);
  // conform to python tuple format (1,)
  if (node->fields.size() == 1) {
    doc << ",";
  }
  return doc << ")";
}

TVM_REGISTER_GLOBAL("tir.AsText")
.set_body_typed<std::string(const ObjectRef&, bool)>(
  [](const ObjectRef& object, bool show_meta) {
    return TIRTextPrinter(show_meta).Print(object).str() +  "\n";
  }
);

}  // namespace tir
}  // namespace tvm
