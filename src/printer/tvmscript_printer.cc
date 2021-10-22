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

#include <tvm/ir/module.h>
#include <tvm/node/serialization.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
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

enum class ExprPrecedence : int {
  /*! \brief Identity(e.g., IntImm, Var) and function call(e.g., floordiv, min) */
  kIdentity = 0,
  /*!
   * \brief Multiplication(*), division(/), and remainder(%)
   * \note floorDiv, floorMod is marked as kIdentity since they are function calls.
   */
  kMultiplicationDivision = 1,
  /*! \brief Addition(+) and subtraction(-) */
  kAdditionSubtraction = 2,
  /*! \brief For relational operators < and <= and > and >= respectively */
  kRelational = 3,
  /*! \brief For equality operators = and != respectively */
  kEquality = 4,
  /*! \brief And(&&) */
  kAnd = 5,
  /*! \brief Or(||) */
  kOr = 6,
  /*! \brief Unknown precedence */
  kUnknown = 7,
};

/*!
 * \brief The printer for TVMScript
 * \details The printer obtain the precedence of the top-level operation when printing each
 *          subexpression to decide whether or not parentheses is needed.
 */
class TVMScriptPrinter : public StmtFunctor<Doc(const Stmt&)>,
                         public ExprFunctor<Doc(const PrimExpr&, ExprPrecedence*)>,
                         public TypeFunctor<Doc(const Type&)> {
 public:
  explicit TVMScriptPrinter(const String& tir_prefix, bool show_meta,
                            runtime::TypedPackedFunc<std::string(Stmt)> annotate = nullptr)
      : tir_prefix_(tir_prefix),
        show_meta_(show_meta),
        annotate_(std::move(annotate)),
        meta_collector_(&meta_) {}

  /*!
   * \brief Print the node.
   * \param node The node to be printed.
   * \param out_precedence The operator precedence of node if it's a PrimExpr,
   *        so we can simplify the bracket.
   */
  TVM_DLL Doc Print(const ObjectRef& node);

 protected:
  /*! \brief The tir prefix */
  String tir_prefix_;
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
  std::unordered_set<const VarNode*> var_not_in_headers_;
  /*!
   * \brief buffer collector
   *        (buffer defined in BufferMap, BufferAllocation and MatchBufferRegion)
   */
  std::unordered_set<const BufferNode*> buf_not_in_headers_;
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
  std::vector<For> simple_loop_stack_;
  /*! \brief the maps from loop_vars to the loops */
  std::unordered_map<const VarNode*, For> loop_var_map_;
  /*!
   * \brief simple block vars remap from loop vars
   * simple_remap requires:
   * 1. block var iter type is kDataPar or kCommReduce
   * 2. value is a single Var, which is a loop_var outside the block
   * 3. The iter range is equal to loop range
   */
  std::vector<std::pair<IterVar, PrimExpr>> block_var_remaps_;

  Doc VisitExpr_(const CastNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const VarNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const AddNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const SubNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const MulNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const DivNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const ModNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const FloorDivNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const FloorModNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const MinNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const MaxNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const EQNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const NENode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const LTNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const LENode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const GTNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const GENode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const AndNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const OrNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const NotNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const SelectNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const IntImmNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const FloatImmNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const StringImmNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const ProducerLoadNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const BufferLoadNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const LoadNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const RampNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const BroadcastNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const LetNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const CallNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const ShuffleNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExpr_(const ReduceNode* op, ExprPrecedence* out_precedence) override;
  Doc VisitExprDefault_(const Object* op, ExprPrecedence* out_precedence) override;

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
  Doc VisitStmt_(const WhileNode* op) override;
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
  Doc PrintBlockVar(const IterVar& iter_var, const PrimExpr& value);
  Doc PrintBlockVarRemaps();
  Doc PrintBlockVars(const BlockRealizeNode* op);
  Doc PrintBlockAttr(const BlockRealizeNode* op);
  Doc PrintBlockBody(const BlockNode* op);
  virtual Doc PrintBlockName(const BlockNode* block_op);
  Doc PrintBufferRegion(const BufferRegionNode* op);
  Doc PrintMatchBufferRegion(const MatchBufferRegionNode* op);
  Doc PrintAnnotations(const Map<String, ObjectRef>& annotations);
  static Doc PrintString(const StringObj* op) { return Doc::StrLiteral(op->data); }

  Doc GetUniqueName(std::string prefix);
  Doc AllocVar(const Var& var);
  Doc AllocBuf(const Buffer& buffer);
  void TryDeallocVar(const Var& var);
  bool ContainsOptionalInfo(const Stmt& stmt);

  /*! Helper functions for loop printing. */
  /*!
   * \brief Print a single for loop
   * \param loop The for loop to be printed
   */
  virtual Doc PrintLoop(const For& loop);
  /*! \brief Print all simple loops in stack into one line using tir_prefix_.grid(). */
  Doc PrintLoopStack();
  /*!
   * \brief Print all simple loops in stack into one line using tir_prefix_.grid().
   * \param for_op the for node to be checked
   */
  bool IsSimpleLoop(const ForNode* for_op) {
    return for_op->kind == ForKind::kSerial && for_op->annotations.empty() &&
           is_zero(for_op->min) && !ContainsOptionalInfo(GetRef<Stmt>(for_op));
  }

  /*!
   * \brief Print additional info about expr in comment.
   * \param expr The expression.
   */
  Doc PrintOptionalInfo(const Stmt& stmt) {
    Doc doc;
    // default annotations
    if (ContainsOptionalInfo(stmt)) {
      std::string annotated_stmt = annotate_(stmt);
      doc << "# " << annotated_stmt << Doc::NewLine();
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
  Doc PrintConstScalar(DataType dtype, const T* data) const {
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
      doc << tir_prefix_ << "." << runtime::DLDataType2String(dtype) << "(" << Doc::Text(os.str())
          << ")";
    }
    return doc;
  }
};

Doc TVMScriptPrinter::GetUniqueName(std::string prefix) {
  std::replace(prefix.begin(), prefix.end(), '.', '_');
  std::string unique_prefix = prefix;
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end() && it->second >= 0) {
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
  bool print_factor_explicitly = false;
  doc << ", dtype=" << PrintDType(buf->dtype);
  if (memo_var_.find(buf->data) != memo_var_.end()) {
    doc << ", data=" << Print(buf->data);
  } else {
    // implicitly define data
    memo_var_[buf->data] = Doc::Text(memo_buf_[buf].str() + ".data");
    var_not_in_headers_.insert(buf->data.get());
  }
  if (!buf->strides.empty()) {
    doc << ", strides=" << Print(buf->strides);
  }
  if (buf->elem_offset->IsInstance<VarNode>()) {
    Var elem_offset = Downcast<Var>(buf->elem_offset);
    if (memo_var_.find(elem_offset) != memo_var_.end()) {
      doc << ", elem_offset=" << Print(buf->elem_offset);
    } else {
      // implicitly define elem_offset
      memo_var_[elem_offset] = Doc::Text(memo_buf_[buf].str() + ".elem_offset");
      var_not_in_headers_.insert(elem_offset.get());
      print_factor_explicitly = true;
    }
  } else if (buf->elem_offset->IsInstance<IntImmNode>()) {
    IntImm elem_offset = Downcast<IntImm>(buf->elem_offset);
    if (elem_offset->value != 0) {
      doc << ", elem_offset=" << Print(buf->elem_offset);
    }
  }
  if (buf.scope() != "global") {
    doc << ", scope=" << Doc::StrLiteral(buf.scope());
  }
  if (buf->data_alignment != runtime::kAllocAlignment) {
    doc << ", align=" << buf->data_alignment;
  }
  if (buf->offset_factor != 1 || print_factor_explicitly) {
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

/*!
 * \brief Check if any optional information exists in annotate_ for
 * a given Stmt.
 * \param stmt The statement.
 */
bool TVMScriptPrinter::ContainsOptionalInfo(const Stmt& stmt) {
  if (annotate_ == nullptr) return false;
  return !annotate_(stmt).empty();
}

/*!
 * \brief Try to dealloc vars out of space and leave the index to coming vars.
 * \note It is not a necessary step.
 */
void TVMScriptPrinter::TryDeallocVar(const Var& var) {
  auto it = memo_var_.find(var);
  ICHECK(it != memo_var_.end());
  std::string print_name = it->second.str();

  std::string name_hint = var->name_hint.operator std::string();
  if (name_hint.length() == 0 || !std::isalpha(name_hint[0])) {
    name_hint = "v" + name_hint;
  }
  std::replace(name_hint.begin(), name_hint.end(), '.', '_');

  auto it2 = name_alloc_map_.find(name_hint);
  // Skip it if we can not find the name_hint in name_alloc_map_.
  if (it2 == name_alloc_map_.end()) return;
  if (it2->second > 0) {
    name_hint = name_hint + '_' + std::to_string(it2->second);
  }
  // Skip it if the name_hint is not equal to how it should be printed.
  if (name_hint != print_name) return;
  // Free the conresponding name_alloc_map_ index
  --it2->second;
}

Doc TVMScriptPrinter::PrintMatchBufferRegion(const MatchBufferRegionNode* op) {
  const Buffer& buf = op->buffer;
  buf_not_in_headers_.insert(buf.get());

  Doc doc = Print(op->buffer) << " = " << tir_prefix_ << ".match_buffer(" << Print(op->source)
                              << ", " << memo_buf_decl_[op->buffer] << ")";
  return doc;
}

Doc TVMScriptPrinter::Print(const ObjectRef& node) {
  if (!node.defined()) return Doc::Text("None");
  if (node->IsInstance<StmtNode>()) {
    return PrintOptionalInfo(Downcast<Stmt>(node)) << VisitStmt(Downcast<Stmt>(node));
  } else if (node->IsInstance<PrimExprNode>()) {
    ExprPrecedence t = ExprPrecedence::kUnknown;
    return VisitExpr(Downcast<PrimExpr>(node), &t);
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

Doc TVMScriptPrinter::VisitExprDefault_(const Object* op, ExprPrecedence* out_precedence) {
  LOG(FATAL) << "Do not know how to print " << op->GetTypeKey();
  return Doc();
}

Doc TVMScriptPrinter::VisitStmtDefault_(const Object* op) {
  LOG(FATAL) << "Do not know how to print " << op->GetTypeKey();
  return Doc();
}

Doc TVMScriptPrinter::VisitExpr_(const IntImmNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  return PrintConstScalar<int64_t>(op->dtype, &(op->value));
}

Doc TVMScriptPrinter::VisitExpr_(const FloatImmNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  return PrintConstScalar<double>(op->dtype, &(op->value));
}

Doc TVMScriptPrinter::VisitExpr_(const StringImmNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  return Doc::StrLiteral(op->value);
}

Doc TVMScriptPrinter::VisitExpr_(const CastNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".cast(" << Print(op->value) << ", " << PrintDType(op->dtype) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const VarNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  const Var& var = GetRef<Var>(op);
  return meta_.InMeta(var) ? meta_.GetMetaNode(var) : AllocVar(GetRef<Var>(op));
}

#define TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(OpName, OpString, OpPrecedence)            \
  Doc TVMScriptPrinter::VisitExpr_(const OpName* op, ExprPrecedence* out_precedence) { \
    Doc doc;                                                                           \
    ExprPrecedence lhs_precedence = ExprPrecedence::kUnknown;                          \
    ExprPrecedence rhs_precedence = ExprPrecedence::kUnknown;                          \
    /* Get children expr out_precedence */                                             \
    Doc lhs_doc = VisitExpr(op->a, &lhs_precedence);                                   \
    Doc rhs_doc = VisitExpr(op->b, &rhs_precedence);                                   \
    ICHECK(lhs_precedence != ExprPrecedence::kUnknown);                                \
    ICHECK(rhs_precedence != ExprPrecedence::kUnknown);                                \
    /* Update out_precedence of current node. */                                       \
    *out_precedence = OpPrecedence;                                                    \
    if (lhs_precedence > OpPrecedence) {                                               \
      doc << "(" << lhs_doc << ")";                                                    \
    } else {                                                                           \
      doc << lhs_doc;                                                                  \
    }                                                                                  \
    doc << OpString;                                                                   \
    if (rhs_precedence >= OpPrecedence) {                                              \
      doc << "(" << rhs_doc << ")";                                                    \
    } else {                                                                           \
      doc << rhs_doc;                                                                  \
    }                                                                                  \
    return doc;                                                                        \
  }

TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(MulNode, " * ", ExprPrecedence::kMultiplicationDivision)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(DivNode, " / ", ExprPrecedence::kMultiplicationDivision)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(ModNode, " % ", ExprPrecedence::kMultiplicationDivision)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(AddNode, " + ", ExprPrecedence::kAdditionSubtraction)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(SubNode, " - ", ExprPrecedence::kAdditionSubtraction)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(LTNode, " < ", ExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(LENode, " <= ", ExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(GTNode, " > ", ExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(GENode, " >= ", ExprPrecedence::kRelational)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(EQNode, " == ", ExprPrecedence::kEquality)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(NENode, " != ", ExprPrecedence::kEquality)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(AndNode, " and ", ExprPrecedence::kAnd)
TVM_DECLARE_TVMSCRIPT_PRINTER_BINOP(OrNode, " or ", ExprPrecedence::kOr)

Doc TVMScriptPrinter::VisitExpr_(const FloorDivNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".floordiv(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const FloorModNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".floormod(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const MinNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".min(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const MaxNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".max(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const NotNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << "not(" << Print(op->a) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const SelectNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".Select(" << Print(op->condition) << ", " << Print(op->true_value) << ", "
      << Print(op->false_value) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const ProducerLoadNode* op, ExprPrecedence* out_precedence) {
  LOG(FATAL) << "Cannot print a tir.ProducerLoad as it is not valid in TIR Primfuncs. You need to "
                "lower this function first.";
  return Doc();
}

Doc TVMScriptPrinter::VisitExpr_(const BufferLoadNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  if (op->indices.size() == 0) {
    doc << Print(op->buffer) << "[()]";
  } else {
    doc << Print(op->buffer) << Print(op->indices);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const LoadNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  if (op->dtype == DataType::Float(32) && is_one(op->predicate) &&
      op->buffer_var->dtype == DataType::Float(32)) {
    doc << Print(op->buffer_var) << "[" << Print(op->index) << "]";
  } else {
    doc << tir_prefix_ << ".load(" << PrintDType(op->dtype) << ", " << Print(op->buffer_var) << ", "
        << Print(op->index);
    if (!is_one(op->predicate) || op->dtype.lanes() != 1) {
      doc << ", " << Print(op->predicate);
    }
    doc << ")";
  }
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const RampNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".ramp(" << Print(op->base) << ", " << Print(op->stride) << ", "
      << op->lanes << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const BroadcastNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".broadcast(" << Print(op->value) << ", " << op->lanes << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const LetNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".let(" << Print(op->var) << ", " << Print(op->value) << ", "
      << Print(op->body) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const CallNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  if (auto* ptr_op = op->op.as<OpNode>()) {
    std::string name = ptr_op->name;
    if (name.find("tir.") == 0) {
      name = tir_prefix_ + "." + name.substr(4);
    }
    doc << name << "(";
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

Doc TVMScriptPrinter::VisitExpr_(const ShuffleNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".shuffle(" << Print(op->vectors) << ", " << Print(op->indices) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitExpr_(const ReduceNode* op, ExprPrecedence* out_precedence) {
  *out_precedence = ExprPrecedence::kIdentity;
  Doc doc;
  doc << tir_prefix_ << ".reduce(" << Print(op->combiner) << ", " << Print(op->source) << ", "
      << Print(op->axis) << ", " << op->value_index << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const LetStmtNode* op) {
  Doc doc;
  if (current_num_ != num_child_ - 1) {
    doc << "with " << tir_prefix_ << ".let(" << Print(op->var) << ", " << Print(op->value) << "):";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    if (memo_var_.find(op->var) == memo_var_.end()) var_not_in_headers_.insert(op->var.get());
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
        doc << "with " << tir_prefix_ << ".realize(" << Print(realize->buffer)
            << Print(realize->bounds) << ", " << Print(op->value);
        if (!is_one(realize->condition)) {
          doc << ", " << Print(realize->condition);
        }
        doc << "):" << Doc::Indent(4, Doc::NewLine() << PrintBody(realize->body));
      } else {
        doc << tir_prefix_ << ".realize(" << Print(realize->buffer) << Print(realize->bounds)
            << ", " << Print(op->value);
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
    var_not_in_headers_.insert(iter_var->var.get());
    var_env_map_[iter_var->var] = iter_var->thread_tag;
    if (current_num_ != num_child_ - 1) {
      doc << "with " << tir_prefix_ << ".launch_thread(" << Print(iter_var->var) << ", "
          << Print(op->value) << "):";
      doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
    } else {
      doc << tir_prefix_ << ".launch_thread(" << Print(iter_var->var) << ", " << Print(op->value)
          << ")";
      doc << Doc::NewLine() << PrintBody(op->body);
    }
    TryDeallocVar(iter_var->var);
    return doc;
  }
  // default
  if (current_num_ != num_child_ - 1) {
    doc << "with " << tir_prefix_ << ".attr(" << Print(op->node) << ", "
        << Doc::StrLiteral(op->attr_key) << ", " << Print(op->value) << "):";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    doc << tir_prefix_ << ".attr(" << Print(op->node) << ", " << Doc::StrLiteral(op->attr_key)
        << ", " << Print(op->value) << ")";
    doc << Doc::NewLine() << PrintBody(op->body);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const AssertStmtNode* op) {
  Doc doc;
  if (current_num_ != num_child_ - 1) {
    doc << "with " << tir_prefix_ << ".Assert(" << Print(op->condition) << ", "
        << Print(op->message) << "):";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    doc << "assert " << Print(op->condition) << ", " << Print(op->message);
    doc << Doc::NewLine() << PrintBody(op->body);
  }
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const StoreNode* op) {
  Doc doc;
  doc << tir_prefix_ << ".store(" << Print(op->buffer_var) << ", " << Print(op->index) << ", "
      << Print(op->value) << ", " << Print(op->predicate) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const BufferRealizeNode* op) {
  LOG(FATAL)
      << "TVM Script Printer Internal Error: All the BufferRealize should be folded with Attr";
  return Doc();
}

Doc TVMScriptPrinter::VisitStmt_(const AllocateNode* op) {
  var_not_in_headers_.insert(op->buffer_var.get());
  Doc doc;
  auto storage_scope = GetPtrStorageScope(op->buffer_var);
  if (current_num_ != num_child_ - 1) {
    doc << "with " << tir_prefix_ << ".allocate(" << Print(op->extents) << ", "
        << PrintDType(op->dtype) << ", " << Print(storage_scope);
    if (!is_one(op->condition)) {
      doc << ", " << Print(op->condition);
    }
    if (!op->annotations.empty()) {
      doc << ", annotations={";
      doc << PrintAnnotations(op->annotations);
      doc << "}";
    }
    doc << ") as " << Print(op->buffer_var) << ":";
    doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  } else {
    doc << Print(op->buffer_var) << " = " << tir_prefix_ << ".allocate(" << Print(op->extents)
        << ", " << PrintDType(op->dtype) << ", " << Print(storage_scope);
    if (!is_one(op->condition)) {
      doc << ", " << Print(op->condition);
    }
    if (!op->annotations.empty()) {
      doc << ", annotations={";
      doc << PrintAnnotations(op->annotations);
      doc << "}";
    }
    doc << ")" << Doc::NewLine() << PrintBody(op->body);
  }
  TryDeallocVar(op->buffer_var);
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
  doc << tir_prefix_ << ".evaluate(" << Print(op->value) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const ForNode* op) {
  Doc doc;
  var_not_in_headers_.insert(op->loop_var.get());
  loop_var_map_[op->loop_var.get()] = GetRef<For>(op);
  const auto* body = op->body.as<ForNode>();
  bool simple_loop = IsSimpleLoop(op);
  if (simple_loop) simple_loop_stack_.push_back(GetRef<For>(op));
  // It is a loop that can be compressed, let the loops below print it out
  if (simple_loop && body != nullptr && IsSimpleLoop(body)) {
    doc << Print(GetRef<For>(body));
    TryDeallocVar(op->loop_var);
    loop_var_map_.erase(op->loop_var.get());
    return doc;
  }
  // It is a loop that can not be compressed
  bool print_above = !simple_loop_stack_.empty();
  // print loops above if needed
  if (print_above) {
    doc << PrintLoopStack();
    simple_loop_stack_.clear();
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
  TryDeallocVar(op->loop_var);
  loop_var_map_.erase(op->loop_var.get());
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const PrefetchNode* op) {
  Doc doc;
  doc << tir_prefix_ << ".prefetch(" << Print(op->buffer) << ", " << Print(op->bounds) << ")";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const WhileNode* op) {
  Doc doc;
  doc << "while " << Print(op->condition) << ":";
  doc << Doc::Indent(4, Doc::NewLine() << PrintBody(op->body));
  return doc;
}

Doc TVMScriptPrinter::VisitType_(const PrimTypeNode* node) {
  Doc doc;
  doc << tir_prefix_ << "." << runtime::DLDataType2String(node->dtype);
  return doc;
}

Doc TVMScriptPrinter::VisitType_(const PointerTypeNode* node) {
  Doc doc;
  doc << tir_prefix_ << ".Ptr[";
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
    return Doc::Text(tir_prefix_ + ".Tuple[") << Doc::Concat(fields) << "]";
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

/*! Helper functions for block printing. */
Doc TVMScriptPrinter::PrintBlockVar(const IterVar& iter_var, const PrimExpr& value) {
  Doc doc;
  doc << Print(iter_var->var) << " = " << tir_prefix_ << ".axis.";
  switch (iter_var->iter_type) {
    case kDataPar:
      doc << "spatial";
      break;
    case kCommReduce:
      doc << "reduce";
      break;
    case kOrdered:
      doc << "scan";
      break;
    case kOpaque:
      doc << "opaque";
      break;
    default:
      LOG(FATAL) << "Unknown block var iter type: " << iter_var->iter_type;
      break;
  }
  doc << "(";
  const Range& dom = iter_var->dom;
  if (is_zero(dom->min)) {
    doc << Print(dom->extent);
  } else {
    doc << "(" << Print(dom->min) << ", " << Print(dom->min + dom->extent) << ")";
  }
  doc << ", " << Print(value) << ")";
  return doc;
}

Doc TVMScriptPrinter::PrintBlockVarRemaps() {
  ICHECK(!block_var_remaps_.empty());
  if (block_var_remaps_.size() == 1) {
    const IterVar& iter_var = block_var_remaps_[0].first;
    const PrimExpr& value = block_var_remaps_[0].second;
    return PrintBlockVar(iter_var, value);
  }
  Doc doc;
  std::vector<Doc> iter_vars, iter_values;
  std::string iter_type;
  for (const auto& pair : block_var_remaps_) {
    const IterVar& iter_var = pair.first;
    const PrimExpr& value = pair.second;
    iter_vars.push_back(Print(iter_var->var));
    iter_values.push_back(Print(value));
    if (iter_var->iter_type == kDataPar) {
      iter_type += "S";
    } else if (iter_var->iter_type == kCommReduce) {
      iter_type += "R";
    } else {
      ICHECK(false);
    }
  }
  doc << PrintSep(iter_vars, Doc::Text(", ")) << " = " << tir_prefix_ << ".axis.remap("
      << Doc::StrLiteral(iter_type) << ", [" << PrintSep(iter_values, Doc::Text(", ")) << "])";
  return doc;
}

Doc TVMScriptPrinter::PrintBlockVars(const BlockRealizeNode* op) {
  Doc doc;
  const auto* block_op = op->block.as<BlockNode>();
  ICHECK_EQ(block_op->iter_vars.size(), op->iter_values.size());
  tir::ExprDeepEqual expr_equal;

  auto is_simple_remap = [this, &expr_equal](const IterVar& iter_var,
                                             const PrimExpr& value) -> bool {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) return false;
    if (!value->IsInstance<VarNode>()) return false;
    const Var& var = Downcast<Var>(value);
    auto it = loop_var_map_.find(var.get());
    return it != loop_var_map_.end() && expr_equal(it->second->min, iter_var->dom->min) &&
           expr_equal(it->second->extent, iter_var->dom->extent);
  };

  for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
    const IterVar& iter_var = block_op->iter_vars[i];
    const PrimExpr& value = op->iter_values[i];
    var_not_in_headers_.insert(iter_var->var.get());
    if (is_simple_remap(iter_var, value)) {
      block_var_remaps_.push_back(std::make_pair(iter_var, value));
    } else {
      if (!block_var_remaps_.empty()) {
        doc << Doc::NewLine() << PrintBlockVarRemaps();
        block_var_remaps_.clear();
      }
      doc << Doc::NewLine() << PrintBlockVar(iter_var, value);
    }
  }
  if (!block_var_remaps_.empty()) {
    doc << Doc::NewLine() << PrintBlockVarRemaps();
    block_var_remaps_.clear();
  }
  return doc;
}

Doc TVMScriptPrinter::PrintBlockAttr(const BlockRealizeNode* op) {
  const auto* block_op = op->block.as<BlockNode>();
  Doc block_attr_doc;
  // print predicate, binding, read/write tensor region, annotations
  if (!is_one(op->predicate)) {
    block_attr_doc << Doc::NewLine() << tir_prefix_ << ".where(" << Print(op->predicate) << ")";
  }
  block_attr_doc << Doc::NewLine() << tir_prefix_ << ".reads(" << Print(block_op->reads) << ")";
  block_attr_doc << Doc::NewLine() << tir_prefix_ << ".writes(" << Print(block_op->writes) << ")";
  if (!block_op->annotations.empty()) {
    block_attr_doc << Doc::NewLine() << tir_prefix_ << ".block_attr({";
    block_attr_doc << PrintAnnotations(block_op->annotations);
    block_attr_doc << "})";
  }
  return block_attr_doc;
}

Doc TVMScriptPrinter::PrintBlockBody(const BlockNode* op) {
  Doc body;
  for (const auto& alloc_buf : op->alloc_buffers) {
    buf_not_in_headers_.insert(alloc_buf.get());
    body << Print(alloc_buf) << " = " << tir_prefix_ << ".alloc_buffer("
         << memo_buf_decl_[alloc_buf] << ")" << Doc::NewLine();
  }
  for (const auto& match_buf : op->match_buffers) {
    body << Print(match_buf) << Doc::NewLine();
  }
  if (op->init.defined()) {
    Doc init_block;
    init_block << "with " << tir_prefix_ << ".init():";
    init_block << Doc::Indent(4, Doc::NewLine() << PrintBody(op->init.value()));
    body << init_block << Doc::NewLine();
  }
  body << PrintBody(op->body);
  return body;
}

/*!
 * \brief Print the name of a block
 * \param block_op The block node to be printed
 */
Doc TVMScriptPrinter::PrintBlockName(const BlockNode* block_op) {
  Doc doc;
  doc << "with " << tir_prefix_ << ".block(";
  if (!block_op->name_hint.empty()) {
    doc << Doc::StrLiteral(block_op->name_hint);
  }
  doc << "):";
  return doc;
}

Doc TVMScriptPrinter::VisitStmt_(const BlockRealizeNode* op) {
  const auto* block_op = op->block.as<BlockNode>();
  // print block name and block vars
  Doc doc = PrintBlockName(block_op);
  Doc block_var = PrintBlockVars(op);
  // print predicate, binding, read/write tensor region, annotations
  Doc block_attr_doc = PrintBlockAttr(op);
  // print body
  Doc body = PrintBlockBody(block_op);
  doc << Doc::Indent(4, block_var << block_attr_doc << Doc::NewLine() << body);
  for (const auto& iter_var : block_op->iter_vars) {
    TryDeallocVar(iter_var->var);
  }
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
  doc << "@tvm.script.ir_module" << Doc::NewLine();
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
  var_not_in_headers_.clear();
  buf_not_in_headers_.clear();
  // print signature
  Doc doc;
  doc << "@" << tir_prefix_ << ".prim_func" << Doc::NewLine();
  doc << "def " << (func2var_.find(op) == func2var_.end() ? "func" : func2var_[op]->name_hint)
      << "(";
  std::vector<Doc> params;
  for (const auto& param : op->params) {
    var_not_in_headers_.insert(param.get());
    params.push_back(Print(param) << ": " << Print(GetType(param)));
  }
  doc << PrintSep(params, Doc::Text(", ")) << ") -> " << Print(primFunc->ret_type) << ":";

  Doc body = Doc::NewLine();
  // print buffer_bind
  for (const auto& param : op->params) {
    auto it = op->buffer_map.find(param);
    if (it == op->buffer_map.end()) continue;
    buf_not_in_headers_.insert((*it).second.get());
    body << Print((*it).second) << " = " << tir_prefix_ << ".match_buffer(";
    body << Print((*it).first) << ", " << memo_buf_decl_[(*it).second];
    body << ")" << Doc::NewLine();
  }
  // print comm_reducer
  for (const auto& it : memo_reducer_) {
    body << it.second << " = .comm_reducer(";
    var_not_in_headers_.insert(it.first->lhs[0].get());
    var_not_in_headers_.insert(it.first->rhs[0].get());
    body << "lambda " << Print(it.first->lhs[0]) << ", " << Print(it.first->rhs[0]) << ": "
         << Print(it.first->result[0]) << ", " << Print(it.first->identity_element[0]);
    body << ")" << Doc::NewLine();
  }
  // print body
  body << "# body" << Doc::NewLine();
  if (op->body->IsInstance<BlockRealizeNode>() &&
      op->body.as<BlockRealizeNode>()->iter_values.empty()) {
    // Skip print root block
    body << "# with " << tir_prefix_ << ".block(\"root\")" << Doc::NewLine();
    const BlockNode* block = op->body.as<BlockRealizeNode>()->block.get();
    body << PrintBlockBody(block);
  } else {
    body << PrintBody(op->body);
  }
  // print func attrs
  Doc header_attr;
  if (primFunc->attrs.defined()) {
    header_attr << Doc::NewLine() << "# function attr dict" << Doc::NewLine() << tir_prefix_
                << ".func_attr({";
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
    if (buf_not_in_headers_.find(it.first.get()) == buf_not_in_headers_.end()) {
      bufs.push_back(it.first.get());
    }
  }
  if (!bufs.empty()) {
    header_buf << Doc::NewLine() << "# buffer definition";
    std::sort(bufs.begin(), bufs.end(), [&](const BufferNode* a, const BufferNode* b) {
      return memo_buf_[GetRef<Buffer>(a)].str() < memo_buf_[GetRef<Buffer>(b)].str();
    });
    for (const auto& buf : bufs) {
      header_buf << Doc::NewLine() << Print(GetRef<Buffer>(buf)) << " = " << tir_prefix_
                 << ".buffer_decl(";
      header_buf << memo_buf_decl_[GetRef<Buffer>(buf)] << ")";
    }
  }
  // print var declaration
  Doc header_var;
  std::vector<const VarNode*> vars;
  for (const auto& it : memo_var_) {
    if (var_not_in_headers_.find(it.first.get()) == var_not_in_headers_.end()) {
      vars.push_back(it.first.get());
    }
  }
  if (!var_env_map_.empty()) {
    header_var << Doc::NewLine() << "# var definition";
    for (const auto& it : var_env_map_) {
      header_var << Doc::NewLine() << Print(it.first) << " = " << tir_prefix_ << ".env_thread("
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
        header_var << Doc::NewLine() << Print(GetRef<Var>(var)) << " = " << tir_prefix_
                   << ".buffer_var(";
        header_var << PrintDType(prim_type->dtype) << ", "
                   << Doc::StrLiteral(ptr_type->storage_scope) << ")";
      } else {
        header_var << Doc::NewLine() << Print(GetRef<Var>(var)) << " = " << tir_prefix_ << ".var(";
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
  doc << tir_prefix_ << ".iter_var(" << Print(op->var);
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
        doc << Print(range->min) << " : " << Print(range->min + range->extent);
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
  res << "for " << Print(loop->loop_var) << " in " << tir_prefix_
      << "." + std::string(ForKind2String(loop->kind)) + "(" << Print(loop->min) << ", "
      << Print(loop->min + loop->extent);
  if (loop->thread_binding.defined()) {
    res << ", thread=";
    res << Print(loop->thread_binding.value()->thread_tag);
  }
  if (!loop->annotations.empty()) {
    res << ", annotations={";
    res << PrintAnnotations(loop->annotations);
    res << "}";
  }
  res << "):";
  return res;
}

Doc TVMScriptPrinter::PrintLoopStack() {
  Doc res;
  if (simple_loop_stack_.size() == 1) {
    res << PrintLoop(simple_loop_stack_[0]);
  } else if (simple_loop_stack_.size() > 1) {
    std::vector<Doc> vars, extents;
    for (const auto& loop : simple_loop_stack_) {
      vars.push_back(Print(loop->loop_var));
      extents.push_back(Print(loop->extent));
    }
    res << "for " << PrintSep(vars, Doc::Text(", ")) << " in " << tir_prefix_ << ".grid("
        << PrintSep(extents, Doc::Text(", ")) << "):";
  }
  return res;
}

/*!
 * \brief The printer for TVMScript with diagnostic
 * \details The printer obtain the precedence of the top-level operation when printing each
 *          subexpression to decide whether or not parentheses is needed.
 */
class TVMScriptPrinterWithDiagnostic : public TVMScriptPrinter {
 public:
  explicit TVMScriptPrinterWithDiagnostic(const String& tir_prefix, bool show_meta,
                                          runtime::TypedPackedFunc<std::string(Stmt)> annotate)
      : TVMScriptPrinter(tir_prefix, show_meta, annotate) {}

 protected:
  Doc PrintBlockName(const BlockNode* block_op) override;
  Doc PrintUnderline(const Stmt& stmt, int length);
  Doc PrintLoop(const For& loop) override;
};

Doc TVMScriptPrinterWithDiagnostic::PrintBlockName(const BlockNode* block_op) {
  Doc doc = PrintOptionalInfo(GetRef<Stmt>(block_op));
  Doc block_name = TVMScriptPrinter::PrintBlockName(block_op);
  doc << block_name << PrintUnderline(GetRef<Stmt>(block_op), block_name.str().size());
  return doc;
}

Doc TVMScriptPrinterWithDiagnostic::PrintUnderline(const Stmt& stmt, int length) {
  Doc doc;
  // annotation
  if (ContainsOptionalInfo(stmt)) {
    String underline = std::string(length, '^');
    doc << Doc::NewLine() << underline;
  }
  return doc;
}

Doc TVMScriptPrinterWithDiagnostic::PrintLoop(const For& loop) {
  Doc res = TVMScriptPrinter::PrintLoop(loop);
  res << PrintUnderline(loop, res.str().size());
  return res;
}

String AsTVMScript(const ObjectRef& mod, const String& tir_prefix, bool show_meta) {
  ICHECK(mod->IsInstance<PrimFuncNode>() || mod->IsInstance<IRModuleNode>());
  return TVMScriptPrinter(tir_prefix, show_meta).Print(mod).str() + "\n";
}

TVM_REGISTER_GLOBAL("script.AsTVMScript").set_body_typed(AsTVMScript);

String AsTVMScriptWithDiagnostic(const ObjectRef& mod, const String& tir_prefix, bool show_meta,
                                 runtime::TypedPackedFunc<std::string(Stmt)> annotate) {
  ICHECK(mod->IsInstance<PrimFuncNode>() || mod->IsInstance<IRModuleNode>());
  return TVMScriptPrinterWithDiagnostic(tir_prefix, show_meta, annotate).Print(mod).str() + "\n";
}

TVM_REGISTER_GLOBAL("script.AsTVMScriptWithDiagnostic").set_body_typed(AsTVMScriptWithDiagnostic);

}  // namespace tir
}  // namespace tvm
