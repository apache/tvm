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
 * \file printer/tvmscript_printer.cc
 * \brief Printer class to print Tensor IR to python syntax script
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/node/serialization.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <utility>

#include "../tir/transforms/ir_utils.h"
#include "doc.h"
#include "meta_data.h"
#include "text_printer.h"

namespace tvm {
namespace tir {

class TVMScriptPrinter : public StmtFunctor<Doc(const Stmt&)>,
                         public ExprFunctor<Doc(const PrimExpr&)>,
                         public TypeFunctor<Doc(const Type&)> {
 public:
  explicit TVMScriptPrinter(bool show_meta,
                            runtime::TypedPackedFunc<std::string(Stmt)> annotate = nullptr)
      : show_meta_(show_meta), annotate_(std::move(annotate)), meta_collector_(&meta_) {}

  /*! \brief Print the node */
  TVM_DLL Doc Print(const ObjectRef& node);

 private:
  /*! \brief whether show meta data */
  bool show_meta_;
  /*! \brief additional comment function */
  runtime::TypedPackedFunc<std::string(Stmt)> annotate_;
  /*! \brief meta data context */
  TextMetaDataContext meta_;
  /*! \brief meta collector */
  MetaCollector meta_collector_;
  /*! \brief map from Function to GlobalVar */
  std::unordered_map<const BaseFuncNode*, GlobalVar> func2var_;
  /*! \brief var collector (var defined by For/Loop/Block) */
  std::unordered_set<const VarNode*> var_not_in_headers;
  /*!
   * \brief buffer collector
   *        (buffer defined in BufferMap, BufferAllocation and MatchBufferRegion)
   */
  std::unordered_set<const BufferNode*> buf_not_in_headers;
  /*! \brief Map from Var to thread env name */
  std::unordered_map<Var, String, ObjectPtrHash, ObjectPtrEqual> var_env_map_;
  /*! \brief Map from Var to Doc */
  std::unordered_map<Var, Doc, ObjectPtrHash, ObjectPtrEqual> memo_var_;
  /*! \brief Map from Buffer to Doc */
  std::unordered_map<Buffer, Doc, ObjectPtrHash, ObjectPtrEqual> memo_buf_;
  /*! \brief Map from Buffer to Declaration Doc */
  std::unordered_map<Buffer, Doc, ObjectPtrHash, ObjectPtrEqual> memo_buf_decl_;
  /*! \brief Map from CommReducer to Doc */
  std::unordered_map<const CommReducerNode*, Doc> memo_reducer_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief number of children of current node's parent */
  int num_child_;
  /*! \brief the number of current node */
  int current_num_;
  /*! \brief loop stack without annotations */
  std::vector<For> loop_stack_;

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
  Doc VisitExpr_(const IntImmNode* op) override;
  Doc VisitExpr_(const FloatImmNode* op) override;
  Doc VisitExpr_(const StringImmNode* op) override;
  Doc VisitExpr_(const ProducerLoadNode* op) override;
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
  Doc VisitStmt_(const IfThenElseNode* op) override;
  Doc VisitStmt_(const SeqStmtNode* op) override;
  Doc VisitStmt_(const ForNode* op) override;
  Doc VisitStmt_(const PrefetchNode* op) override;
  Doc VisitStmt_(const EvaluateNode* op) override;
  Doc VisitStmt_(const BlockRealizeNode* op) override;
  Doc VisitStmtDefault_(const Object* op) override;

  Doc VisitType_(const PrimTypeNode* node) override;
  Doc VisitType_(const PointerTypeNode* node) override;
  Doc VisitType_(const TupleTypeNode* node) override;

  Doc PrintBody(const Stmt& body);
  Doc PrintIRModule(const IRModule& module);
  Doc PrintPrimFunc(const PrimFunc& primFunc);
  Doc PrintIterVar(const IterVarNode* op);
  Doc PrintRange(const RangeNode* op);
  Doc PrintArray(const ArrayNode* op);
  Doc PrintBuffer(const BufferNode* op);
  Doc AllocBufferDeclaration(const Buffer& buf);
  Doc PrintBufferRegion(const BufferRegionNode* op);
  Doc PrintMatchBufferRegion(const MatchBufferRegionNode* op);
  Doc PrintAnnotations(const Map<String, ObjectRef>& annotations);
  static Doc PrintString(const StringObj* op) { return Doc::StrLiteral(op->data); }

  Doc GetUniqueName(std::string prefix);
  Doc AllocVar(const Var& var);
  Doc AllocBuf(const Buffer& buffer);

  /*! Helper functions for loop printing. */
  /*!
   * \brief Print a single for loop
   * \param loop The for loop to be printed
   */
  Doc PrintLoop(const For& loop);
  /*! \brief Print all simple loops in stack into one line using tir.grid(). */
  Doc PrintLoopStack();

  /*!
   * \brief Print additional info about expr in comment.
   * \param expr The expression.
   */
  Doc PrintOptionalInfo(const Stmt& stmt) {
    Doc doc;
    // default annotations
    if (annotate_ != nullptr) {
      std::string annotated_stmt = annotate_(stmt);
      if (!annotated_stmt.empty()) {
        doc << "# " << annotated_stmt << Doc::NewLine();
      }
    }
    return doc;
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

  /*!
   * \brief special method to print out data type
   * \param dtype The data type
   */
  static Doc PrintDType(DataType dtype) {
    return Doc::StrLiteral(runtime::DLDataType2String(dtype));
  }

  /*!
   * \brief special method to print out const scalar
   * \param dtype The data type
   * \param data The pointer to hold the data.
   */
  template <typename T>
  static Doc PrintConstScalar(DataType dtype, const T* data) {
    Doc doc;
    std::ostringstream os;
    if (dtype.is_float() || dtype.is_float16() || dtype.is_bfloat16()) {
      os.precision(17);
    }
    os << data[0];
    if (dtype == DataType::Int(32)) {
      doc << Doc::Text(os.str());
    } else if (dtype == DataType::Bool()) {
      doc << Doc::Text(data[0] ? "True" : "False");
    } else {
      doc << "tir." << runtime::DLDataType2String(dtype) << "(" << Doc::Text(os.str()) << ")";
    }
    return doc;
  }
};

Doc TVMScriptPrinter::GetUniqueName(std::string prefix) {
  std::replace(prefix.begin(), prefix.end(), '.', '_');
  std::string unique_prefix = prefix;
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end()) {
    while (name_alloc_map_.count(unique_prefix = prefix + "_" + std::to_string(++it->second)) > 0) {
    }
  }
  name_alloc_map_[unique_prefix] = 0;
  return Doc::Text(unique_prefix);
}

Doc TVMScriptPrinter::AllocVar(const Var& var) {
  const auto& it = memo_var_.find(var);
  if (it != memo_var_.end()) {
    return it->second;
  }
  std::string name = var->name_hint.operator std::string();
  if (name.length() == 0 || !std::isalpha(name[0])) {
    name = "v" + name;
  }
  Doc val = GetUniqueName(name);
  memo_var_[var] = val;
  return val;
}

Doc TVMScriptPrinter::AllocBufferDeclaration(const Buffer& buf) {
  Doc doc = Print(buf->shape);
  if (!runtime::TypeEqual(buf->dtype, DataType::Float(32))) {
    doc << ", dtype=" << PrintDType(buf->dtype);
  }
  if (memo_var_.find(buf->data) != memo_var_.end()) {
    doc << ", data=" << Print(buf->data);
  } else {
    // implicitly define data
    memo_var_[buf->data] = Doc::Text(memo_buf_[buf].str() + ".data");
    var_not_in_headers.insert(buf->data.get());
  }
  if (!buf->strides.empty()) {
    doc << ", strides=" << Print(buf->strides);
  }
  if (buf->offset_factor != 0 && buf->elem_offset->IsInstance<VarNode>()) {
    Var elem_offset = Downcast<Var>(buf->elem_offset);
    if (memo_var_.find(elem_offset) != memo_var_.end()) {
      doc << ", elem_offset=" << Print(buf->elem_offset);
    } else {
      // implicitly define elem_offset
      memo_var_[elem_offset] = Doc::Text(memo_buf_[buf].str() + ".elem_offset");
      var_not_in_headers.insert(elem_offset.get());
    }
  } else {
    doc << ", elem_offset=" << Print(buf->elem_offset);
  }
  if (buf.scope() != "global") {
    doc << ", scope=" << Doc::StrLiteral(buf.scope());
  }
  if (buf->data_alignment != -1) {
    doc << ", align=" << buf->data_alignment;
  }
  if (buf->offset_factor != 0) {
    doc << ", offset_factor=" << buf->offset_factor;
  }
  if (buf->buffer_type != 1) {
    doc << ", type=" << Doc::StrLiteral("auto");
  }
  return doc;
}

Doc TVMScriptPrinter::AllocBuf(const Buffer& buffer) {
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
  memo_buf_decl_[buffer] = AllocBufferDeclaration(buffer);
  return val;
}

Doc TVMScriptPrinter::PrintMatchBufferRegion(const MatchBufferRegionNode* op) {
  const Buffer& buf = op->buffer;
  buf_not_in_headers.insert(buf.get());

  Doc doc = Print(op->buffer) << " = tir.match_buffer(" << Print(op->source) << ", "
                              << memo_buf_decl_[op->buffer] << ")";
  return doc;
}

Doc TVMScriptPrinter::Print(const ObjectRef& node) {
  if (!node.defined()) return Doc::Text("None");
  if (node->IsInstance<StmtNode>()) {
    return PrintOptionalInfo(Downcast<Stmt>(node)) << VisitStmt(Downcast<Stmt>(node));
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
  } else if (node->IsInstance<BufferNode>()) {
    return PrintBuffer(node.as<BufferNode>());
  } else if (node->IsInstance<StringObj>()) {
    return PrintString(node.as<StringObj>());
  } else if (node->IsInstance<IterVarNode>()) {
    return PrintIterVar(node.as<IterVarNode>());
  } else if (node->IsInstance<RangeNode>()) {
    return PrintRange(node.as<RangeNode>());
  } else if (node->IsInstance<BufferRegionNode>()) {
    return PrintBufferRegion(node.as<BufferRegionNode>());
  } else if (node->IsInstance<MatchBufferRegionNode>()) {
    return PrintMatchBufferRegion(node.as<MatchBufferRegionNode>());
  } else {
    LOG(FATAL) << "Do not know how to print " << node->GetTypeKey();
    return Doc();
  }
}

Doc TVMScriptPrinter::VisitExprDefault_(const Object* op) {
  LOG(FATAL) << "Do not know how to print " << op->GetTypeKey();
  return Doc();
}

Doc TVMScriptPrinter::VisitStmtDefault_(const Object* op) {
  LOG(FATAL) << "Do not know how to print " << op->GetTypeKey();
  return Doc();
}

Doc TVMScriptPrinter::VisitExpr_(const IntImmNode* op) {
  return PrintConstScalar<int64_t>(op->dtype, &(op->value));
}

Doc TVMScriptPrinter::VisitExpr_(const FloatImmNode* op) {
  return PrintConstScalar<double>(op->dtype, &(op->value));
}

Doc TVMScriptPrinter::VisitExpr_(const StringImmNode* op) { return Doc::StrLiteral(op->value); }

Doc TVMScriptPrinter::VisitExpr_(const CastNode* op) {
  Doc doc;
  doc << "tir.cast(" << Print(op->value) << ", " << PrintDType(op->dtype) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const VarNode* op) {
  const Var& var = GetRef<Var>(op);
  return meta_.InMeta(var) ? meta_.GetMetaNode(var) : AllocVar(GetRef<Var>(op));
}

#define TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(OpName, OpString)      \
  Doc TVMScriptPrinter::VisitExpr_(const OpName* op) {             \
    Doc doc;                                                       \
    doc << '(' << Print(op->a) << OpString << Print(op->b) << ")"; \
    return doc;                                                    \
  }

TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(AddNode, " + ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(SubNode, " - ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(MulNode, "*")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(DivNode, " / ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(ModNode, " % ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(EQNode, " == ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(NENode, " != ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(LTNode, " < ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(LENode, " <= ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(GTNode, " > ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(GENode, " >= ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(AndNode, " and ")
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(OrNode, " or ")

Doc TVMScriptPrinter::VisitExpr_(const FloorDivNode* op) {
  Doc doc;
  doc << "tir.floordiv(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const FloorModNode* op) {
  Doc doc;
  doc << "tir.floormod(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const MinNode* op) {
  Doc doc;
  doc << "tir.min(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const MaxNode* op) {
  Doc doc;
  doc << "tir.max(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const NotNode* op) {
  Doc doc;
  doc << "not (" << Print(op->a) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const SelectNode* op) {
  Doc doc;
  doc << "tir.Select(" << Print(op->condition) << ", " << Print(op->true_value) << ", "
      << Print(op->false_value) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const ProducerLoadNode* op) {
  LOG(FATAL) << "Cannot print a tir.ProducerLoad as it is not valid in TIR Primfuncs. You need to "
                "lower this function first.";
  return Doc();
}

Doc TVMScriptPrinter::VisitExpr_(const BufferLoadNode* op) {
  Doc doc;
  if (op->indices.size() == 0) {
    doc << Print(op->buffer) << "[()]";
  } else {
    doc << Print(op->buffer) << Print(op->indices);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const LoadNode* op) {
  Doc doc;
  if (op->dtype == DataType::Float(32) && is_one(op->predicate) &&
      op->buffer_var->dtype == DataType::Float(32)) {
    doc << Print(op->buffer_var) << "[" << Print(op->index) << "]";
  } else {
    doc << "tir.load(" << PrintDType(op->dtype) << ", " << Print(op->buffer_var) << ", "
        << Print(op->index);
    if (!is_one(op->predicate) || op->dtype.lanes() != 1) {
      doc << ", " << Print(op->predicate);
    }
    doc << ")";
  }
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const RampNode* op) {
  Doc doc;
  doc << "tir.ramp(" << Print(op->base) << ", " << Print(op->stride) << ", " << op->lanes << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const BroadcastNode* op) {
  Doc doc;
  doc << "tir.broadcast(" << Print(op->value) << ", " << op->lanes << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const LetNode* op) {
  Doc doc;
  doc << "tir.let(" << Print(op->var) << ", " << Print(op->value) << ", " << Print(op->body) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const CallNode* op) {
  Doc doc;
  if (auto* ptr_op = op->op.as<OpNode>()) {
    doc << Doc::Text(ptr_op->name) << "(";
  } else {
    auto* op_gvar = op->op.as<GlobalVarNode>();
    ICHECK(op_gvar != nullptr);
    doc << Doc::Text(op_gvar->name_hint) << "(";
  }
  std::vector<Doc> args;
  for (const auto& arg : op->args) {
    args.push_back(Print(arg));
  }
  args.push_back(Doc::Text("dtype=") << PrintDType(op->dtype));
  doc << PrintSep(args, Doc::Text(", ")) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const ShuffleNode* op) {
  Doc doc;
  doc << "tir.shuffle(" << Print(op->vectors) << ", " << Print(op->indices) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const ReduceNode* op) {
  Doc doc;
  doc << "tir.reduce(" << Print(op->combiner) << ", " << Print(op->source) << ", "
      << Print(op->axis) << ", " << op->value_index << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const LetStmtNode* op) {
  Doc doc;
  if (current_num_ != num_child_ - 1) {
    doc << "with tir.let(" << Print(op->var) << ", " << Print(op->value) << "):";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    if (memo_var_.find(op->var) == memo_var_.end()) var_not_in_headers.insert(op->var.get());
    doc << Print(op->var) << ": " << Print(GetType(op->var)) << " = " << Print(op->value)
        << Doc::NewLine() << PrintBody(op->body);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const AttrStmtNode* op) {
  Doc doc;
  // merge attr with realize when possible
  if (op->node->IsInstance<BufferNode>() && op->attr_key == "realize_scope" &&
      op->body->IsInstance<BufferRealizeNode>()) {
    const auto* realize = Downcast<BufferRealize>(op->body).get();
    if (realize->buffer.same_as(op->node)) {
      if (current_num_ != num_child_ - 1) {
        doc << "with tir.realize(" << Print(realize->buffer) << Print(realize->bounds) << ", "
            << Print(op->value);
        if (!is_one(realize->condition)) {
          doc << ", " << Print(realize->condition);
        }
        doc << "):" << Doc::Indent(4, Doc::NewLine() << PrintBody(realize->body));
      } else {
        doc << "tir.realize(" << Print(realize->buffer) << Print(realize->bounds) << ", "
            << Print(op->value);
        if (!is_one(realize->condition)) {
          doc << ", " << Print(realize->condition);
        }
        doc << ")" << Doc::NewLine() << PrintBody(realize->body);
      }
      return doc;
    }
  }
  // concise thread env
  if (op->node->IsInstance<IterVarNode>() &&
      (op->attr_key == "thread_extent" || op->attr_key == "virtual_thread")) {
    const auto* iter_var = Downcast<IterVar>(op->node).get();
    var_not_in_headers.insert(iter_var->var.get());
    var_env_map_[iter_var->var] = iter_var->thread_tag;
    if (current_num_ != num_child_ - 1) {
      doc << "with tir.launch_thread(" << Print(iter_var->var) << ", " << Print(op->value) << "):";
      doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
    } else {
      doc << "tir.launch_thread(" << Print(iter_var->var) << ", " << Print(op->value) << ")";
      doc << Doc::NewLine() << PrintBody(op->body);
    }
    return doc;
  }
  // default
  if (current_num_ != num_child_ - 1) {
    doc << "with tir.attr(" << Print(op->node) << ", " << Doc::StrLiteral(op->attr_key) << ", "
        << Print(op->value) << "):";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    doc << "tir.attr(" << Print(op->node) << ", " << Doc::StrLiteral(op->attr_key) << ", "
        << Print(op->value) << ")";
    doc << Doc::NewLine() << PrintBody(op->body);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const AssertStmtNode* op) {
  Doc doc;
  if (current_num_ != num_child_ - 1) {
    doc << "with tir.Assert(" << Print(op->condition) << ", " << Print(op->message) << "):";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    doc << "assert " << Print(op->condition) << ", " << Print(op->message);
    doc << Doc::NewLine() << PrintBody(op->body);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const StoreNode* op) {
  Doc doc;
  doc << "tir.store(" << Print(op->buffer_var) << ", " << Print(op->index) << ", "
      << Print(op->value) << ", " << Print(op->predicate) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const BufferRealizeNode* op) {
  LOG(FATAL)
      << "TVM Script Printer Internal Error: All the BufferRealize should be folded with Attr";
  return Doc();
}

Doc TVMScriptPrinter::VisitStmt_(const AllocateNode* op) {
  var_not_in_headers.insert(op->buffer_var.get());
  Doc doc;
  auto storage_scope = GetPtrStorageScope(op->buffer_var);
  if (current_num_ != num_child_ - 1) {
    doc << "with tir.allocate(" << Print(op->extents) << ", " << PrintDType(op->dtype) << ", "
        << Print(storage_scope);
    if (!is_one(op->condition)) {
      doc << ", " << Print(op->condition);
    }
    doc << ") as " << Print(op->buffer_var) << ":";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    doc << Print(op->buffer_var) << " = tir.allocate(" << Print(op->extents) << ", "
        << PrintDType(op->dtype) << ", " << Print(storage_scope);
    if (!is_one(op->condition)) {
      doc << ", " << Print(op->condition);
    }
    doc << ")" << Doc::NewLine() << PrintBody(op->body);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const IfThenElseNode* op) {
  Doc doc;
  doc << "if " << Print(op->condition) << ":";
  doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->then_case));
  if (!is_one(op->condition) && op->else_case.defined()) {
    doc << Doc::NewLine();
    doc << "else:" << Doc::Indent(4, Doc::NewLine() << PrintBody(op->else_case));
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const SeqStmtNode* op) {
  std::vector<Doc> stmts;
  for (Stmt stmt : op->seq) {
    stmts.push_back(Print(stmt));
  }
  return PrintSep(stmts, Doc::NewLine());
}

Doc TVMScriptPrinter::VisitStmt_(const EvaluateNode* op) {
  Doc doc;
  doc << "tir.evaluate(" << Print(op->value) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const ForNode* op) {
  Doc doc;
  var_not_in_headers.insert(op->loop_var.get());
  const auto* body = op->body.as<ForNode>();
  bool simple_loop = op->kind == ForKind::kSerial && op->annotations.empty() && is_zero(op->min);
  if (simple_loop) loop_stack_.push_back(GetRef<For>(op));
  // It is a loop that can be compressed, let the loops below print it out
  if (simple_loop && body != nullptr) return Print(GetRef<For>(body));
  // It is a loop that can not be compressed
  bool print_above = !loop_stack_.empty();
  // print loops above if needed
  if (print_above) {
    doc << PrintLoopStack();
    loop_stack_.clear();
  }
  if (!simple_loop) {
    // print current loop if needed
    Doc current_loop;
    current_loop << PrintLoop(GetRef<For>(op));
    current_loop << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
    doc << (print_above ? Doc::Indent(4, Doc::NewLine() << current_loop) : current_loop);
  } else {
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const PrefetchNode* op) {
  Doc doc;
  doc << "tir.prefetch(" << Print(op->buffer) << ", " << Print(op->bounds) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitType_(const PrimTypeNode* node) {
  Doc doc;
  doc << "ty." << runtime::DLDataType2String(node->dtype);
  return doc;
}

Doc TVMScriptPrinter::VisitType_(const PointerTypeNode* node) {
  Doc doc;
  doc << "ty.Ptr[";
  if (!node->storage_scope.empty()) {
    doc << node->storage_scope << " ";
  }
  doc << Print(node->element_type) << "]";
  return doc;
}

Doc TVMScriptPrinter::VisitType_(const TupleTypeNode* node) {
  if (node->fields.empty()) {
    return Doc::Text("None");
  } else {
    std::vector<Doc> fields;
    for (Type field : node->fields) {
      fields.push_back(Print(field));
    }
    return Doc::Text("ty.Tuple[") << Doc::Concat(fields) << "]";
  }
}

Doc TVMScriptPrinter::VisitStmt_(const BufferStoreNode* op) {
  Doc doc;
  if (op->indices.size() == 0) {
    doc << Print(op->buffer) << "[()] = " << Print(op->value);
  } else {
    doc << Print(op->buffer) << Print(op->indices) << " = " << Print(op->value);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const BlockRealizeNode* op) {
  const auto* block_op = op->block.as<BlockNode>();
  // print block name and block vars
  Doc doc;
  doc << "with tir.block([";
  std::vector<Doc> block_var_docs;
  for (const auto& iter_var : block_op->iter_vars) {
    Doc block_var_doc;
    if (is_zero(iter_var->dom->min) && iter_var->iter_type == kDataPar) {
      block_var_doc << Print(iter_var->dom->extent);
    } else {
      block_var_doc << "tir.";
      switch (iter_var->iter_type) {
        case kDataPar:
          block_var_doc << "range";
          break;
        case kCommReduce:
          block_var_doc << "reduce_axis";
          break;
        case kOrdered:
          block_var_doc << "scan_axis";
          break;
        case kOpaque:
          block_var_doc << "opaque_axis";
          break;
        default:
          LOG(FATAL) << "Unknown block var iter type: " << iter_var->iter_type;
          break;
      }
      block_var_doc << "(" << Print(iter_var->dom->min) << ", "
                    << Print(iter_var->dom->min + iter_var->dom->extent) << ")";
    }
    block_var_docs.push_back(block_var_doc);
  }
  doc << PrintSep(block_var_docs, Doc::Text(", ")) << "], ";
  doc << Doc::StrLiteral(block_op->name_hint) << ")";
  std::vector<Doc> block_var_names;
  for (const auto& iter_var : block_op->iter_vars) {
    var_not_in_headers.insert(iter_var->var.get());
    block_var_names.push_back(Print(iter_var->var));
  }
  if (!block_var_names.empty()) {
    doc << " as [" << PrintSep(block_var_names, Doc::Text(", ")) << "]";
  }
  doc << ":";
  Doc block_attr_doc;
  // print predicate, binding, read/write tensor region, annotations
  if (!is_one(op->predicate)) {
    block_attr_doc << Doc::NewLine() << "tir.where(" << Print(op->predicate) << ")";
  }
  for (size_t i = 0; i < block_op->iter_vars.size(); ++i)
    block_attr_doc << Doc::NewLine() << "tir.bind(" << Print(block_op->iter_vars[i]->var) << ", "
                   << Print(op->iter_values[i]) << ")";
  block_attr_doc << Doc::NewLine() << "tir.reads(" << Print(block_op->reads) << ")";
  block_attr_doc << Doc::NewLine() << "tir.writes(" << Print(block_op->writes) << ")";
  if (!block_op->annotations.empty()) {
    block_attr_doc << Doc::NewLine() << "tir.block_attr({";
    block_attr_doc << PrintAnnotations(block_op->annotations);
    block_attr_doc << "})";
  }
  // print body
  Doc body;
  body << Doc::NewLine();
  for (const auto& alloc_buf : block_op->alloc_buffers) {
    buf_not_in_headers.insert(alloc_buf.get());
    body << Print(alloc_buf) << " = tir.alloc_buffer(" << memo_buf_decl_[alloc_buf] << ")"
         << Doc::NewLine();
  }
  for (const auto& match_buf : block_op->match_buffers) {
    body << Print(match_buf) << Doc::NewLine();
  }
  if (block_op->init.defined()) {
    Doc init_block;
    init_block << "with tir.init():";
    init_block << Doc::Indent(4, Doc::NewLine() << PrintBody(block_op->init.value()));
    body << init_block << Doc::NewLine();
  }
  body << PrintBody(block_op->body);
  doc << Doc::Indent(4, block_attr_doc << body);
  return doc;
}

Doc TVMScriptPrinter::PrintBody(const Stmt& body) {
  int memo_num_child, memo_current_num;
  std::swap(memo_num_child, num_child_);
  std::swap(memo_current_num, current_num_);

  Doc doc;
  if (body->IsInstance<SeqStmtNode>()) {
    const auto& op = Downcast<SeqStmt>(body);
    num_child_ = op->seq.size();
    current_num_ = 0;
    std::vector<Doc> stmts;
    for (Stmt stmt : op->seq) {
      stmts.push_back(Print(stmt));
      current_num_++;
    }
    doc = PrintSep(stmts, Doc::NewLine());
  } else {
    num_child_ = 1;
    current_num_ = 0;
    doc = Print(body);
  }

  std::swap(memo_num_child, num_child_);
  std::swap(memo_current_num, current_num_);
  return doc;
}

Doc TVMScriptPrinter::PrintIRModule(const IRModule& module) {
  auto* op = module.operator->();
  Doc doc;
  doc << "class Module:";
  for (const auto& x : op->functions) {
    func2var_[x.second.operator->()] = x.first;
  }
  Doc body = Doc::NewLine();
  std::vector<Doc> functions;
  for (auto it = op->functions.begin(); it != op->functions.end(); ++it) {
    if ((*it).second.as<PrimFuncNode>()) {
      functions.push_back(Print((*it).second));
    }
  }
  body << TVMScriptPrinter::PrintSep(functions, Doc::NewLine() << Doc::NewLine());
  body << Doc::NewLine() << DumpMeta();
  doc << Doc::Indent(4, body);
  return doc;
}

Doc TVMScriptPrinter::PrintPrimFunc(const PrimFunc& primFunc) {
  auto* op = primFunc.operator->();
  // clear renaming map
  memo_var_.clear();
  memo_buf_.clear();
  memo_buf_decl_.clear();
  memo_reducer_.clear();
  var_not_in_headers.clear();
  buf_not_in_headers.clear();
  // print signature
  Doc doc;
  doc << "def " << (func2var_.find(op) == func2var_.end() ? "func" : func2var_[op]->name_hint)
      << "(";
  std::vector<Doc> params;
  for (const auto& param : op->params) {
    var_not_in_headers.insert(param.get());
    params.push_back(Print(param) << ": " << Print(GetType(param)));
  }
  doc << PrintSep(params, Doc::Text(", ")) << ") -> " << Print(primFunc->ret_type) << ":";

  Doc body = Doc::NewLine();
  // print buffer_bind
  for (const auto& it : op->buffer_map) {
    buf_not_in_headers.insert(it.second.get());
    body << Print(it.second) << " = tir.match_buffer(";
    body << Print(it.first) << ", " << memo_buf_decl_[it.second];
    body << ")" << Doc::NewLine();
  }
  // print comm_reducer
  for (const auto& it : memo_reducer_) {
    body << it.second << " = tir.comm_reducer(";
    var_not_in_headers.insert(it.first->lhs[0].get());
    var_not_in_headers.insert(it.first->rhs[0].get());
    body << "lambda " << Print(it.first->lhs[0]) << ", " << Print(it.first->rhs[0]) << ": "
         << Print(it.first->result[0]) << ", " << Print(it.first->identity_element[0]);
    body << ")" << Doc::NewLine();
  }
  // print body
  body << "# body" << Doc::NewLine() << PrintBody(op->body);
  // print func attrs
  Doc header_attr;
  if (primFunc->attrs.defined()) {
    header_attr << Doc::NewLine() << "# function attr dict" << Doc::NewLine() << "tir.func_attr({";
    std::vector<Doc> attrs;
    for (const auto& it : op->attrs->dict) {
      attrs.push_back(Doc::StrLiteral(it.first) << ": " << Print(it.second));
    }
    header_attr << PrintSep(attrs, Doc::Text(", ")) << "})";
  }
  // print buffer declarations(buffers not defined by buffer_bind or buffer_allocate)
  Doc header_buf;
  std::vector<const BufferNode*> bufs;
  for (const auto& it : memo_buf_) {
    if (buf_not_in_headers.find(it.first.get()) == buf_not_in_headers.end()) {
      bufs.push_back(it.first.get());
    }
  }
  if (!bufs.empty()) {
    header_buf << Doc::NewLine() << "# buffer definition";
    std::sort(bufs.begin(), bufs.end(), [&](const BufferNode* a, const BufferNode* b) {
      return memo_buf_[GetRef<Buffer>(a)].str() < memo_buf_[GetRef<Buffer>(b)].str();
    });
    for (const auto& buf : bufs) {
      header_buf << Doc::NewLine() << Print(GetRef<Buffer>(buf)) << " = tir.buffer_decl(";
      header_buf << memo_buf_decl_[GetRef<Buffer>(buf)] << ")";
    }
  }
  // print var declaration
  Doc header_var;
  std::vector<const VarNode*> vars;
  for (const auto& it : memo_var_) {
    if (var_not_in_headers.find(it.first.get()) == var_not_in_headers.end()) {
      vars.push_back(it.first.get());
    }
  }
  if (!var_env_map_.empty()) {
    header_var << Doc::NewLine() << "# var definition";
    for (const auto& it : var_env_map_) {
      header_var << Doc::NewLine() << Print(it.first) << " = tir.env_thread("
                 << Doc::StrLiteral(it.second) << ")";
    }
  }
  if (!vars.empty()) {
    std::sort(vars.begin(), vars.end(), [&](const VarNode* a, const VarNode* b) {
      return memo_var_[GetRef<Var>(a)].str() < memo_var_[GetRef<Var>(b)].str();
    });
    for (const auto& var : vars) {
      auto type = GetRef<Var>(var)->type_annotation;
      if (auto* ptr_type = type.as<PointerTypeNode>()) {
        auto* prim_type = ptr_type->element_type.as<PrimTypeNode>();
        ICHECK(prim_type);
        header_var << Doc::NewLine() << Print(GetRef<Var>(var)) << " = tir.buffer_var(";
        header_var << PrintDType(prim_type->dtype) << ", "
                   << Doc::StrLiteral(ptr_type->storage_scope) << ")";
      } else {
        header_var << Doc::NewLine() << Print(GetRef<Var>(var)) << " = tir.var(";
        header_var << PrintDType(var->dtype) << ")";
      }
    }
  }
  doc << Doc::Indent(4, header_attr << header_var << header_buf << body);
  return doc;
}

Doc TVMScriptPrinter::PrintArray(const ArrayNode* op) {
  Doc doc;
  doc << '[';
  for (size_t i = 0; i < op->size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    doc << Print(op->at(i));
  }
  doc << ']';
  return doc;
}

Doc TVMScriptPrinter::PrintIterVar(const IterVarNode* op) {
  Doc doc;
  doc << "tir.iter_var(" << Print(op->var);
  if (op->dom.defined()) {
    doc << ", [" << Print(op->dom) << "], ";
  } else {
    doc << ", None, ";
  }
  doc << Doc::StrLiteral(IterVarType2String(op->iter_type)) << ", ";
  doc << Doc::StrLiteral(op->thread_tag) << ")";
  return doc;
}

Doc TVMScriptPrinter::PrintRange(const RangeNode* op) {
  return Print(op->min) << ":" << Print(op->min + op->extent);
}

Doc TVMScriptPrinter::PrintBuffer(const BufferNode* op) {
  const Buffer& buffer = GetRef<Buffer>(op);
  return meta_.InMeta(buffer) ? meta_.GetMetaNode(buffer) : AllocBuf(buffer);
}

Doc TVMScriptPrinter::PrintBufferRegion(const BufferRegionNode* op) {
  Doc doc;
  if (op->region.size() == 0) {
    doc << Print(op->buffer) << "[()]";
  } else {
    doc << Print(op->buffer) << "[";
    for (size_t i = 0; i < op->region.size(); ++i) {
      if (i != 0) doc << ", ";
      const auto& range = op->region[i];
      if (!is_one(range->extent)) {
        doc << Print(range->min) << ":" << Print(range->min + range->extent);
      } else {
        doc << Print(range->min);
      }
    }
    doc << "]";
  }
  return doc;
}

Doc TVMScriptPrinter::PrintAnnotations(const Map<String, ObjectRef>& annotations) {
  Doc res;
  std::vector<std::pair<String, ObjectRef>> anno_list;
  anno_list.reserve(annotations.size());
  for (const auto& pair : annotations) {
    anno_list.emplace_back(pair);
  }
  sort(anno_list.begin(), anno_list.end());
  for (size_t i = 0; i < anno_list.size(); ++i) {
    if (i != 0) {
      res << ", ";
    }
    res << "\"" << anno_list[i].first << "\":" << Print(anno_list[i].second);
  }
  return res;
}

Doc TVMScriptPrinter::PrintLoop(const For& loop) {
  Doc res;
  res << "for " << Print(loop->loop_var)
      << " in tir." + std::string(ForKind2String(loop->kind)) + "(" << Print(loop->min) << ", "
      << Print(loop->min + loop->extent);
  if (loop->thread_binding.defined()) {
    res << ", thread = ";
    res << Print(loop->thread_binding.value()->thread_tag);
  }
  if (!loop->annotations.empty()) {
    res << ", annotation = {";
    res << PrintAnnotations(loop->annotations);
    res << "}";
  }
  res << "):";
  return res;
}

Doc TVMScriptPrinter::PrintLoopStack() {
  Doc res;
  if (loop_stack_.size() == 1) {
    res << PrintLoop(loop_stack_[0]);
  } else if (loop_stack_.size() > 1) {
    std::vector<Doc> vars, extents;
    for (const auto& loop : loop_stack_) {
      vars.push_back(Print(loop->loop_var));
      extents.push_back(Print(loop->extent));
    }
    res << "for " << PrintSep(vars, Doc::Text(", ")) << " in tir.grid("
        << PrintSep(extents, Doc::Text(", ")) << "):";
  }
  return res;
}

String AsTVMScript(const ObjectRef& mod, bool show_meta) {
  ICHECK(mod->IsInstance<PrimFuncNode>() || mod->IsInstance<IRModuleNode>());
  return "@tvm.script.tir\n" + TVMScriptPrinter(show_meta).Print(mod).str() + "\n";
}

TVM_REGISTER_GLOBAL("script.AsTVMScript").set_body_typed(AsTVMScript);

}  // namespace tir
}  // namespace tvm
