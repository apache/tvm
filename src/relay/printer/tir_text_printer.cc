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
 * \file tir_text_printer.cc
 * \brief Printer to print out the IR text format
 *        that can be parsed by a parser.
 */

#include <tvm/ir/module.h>
#include <tvm/ir/type.h>
#include <tvm/ir/type_functor.h>
#include <tvm/node/serialization.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <algorithm>
#include <string>

#include "../../tir/transforms/ir_utils.h"
#include "doc.h"
#include "meta_data.h"
#include "text_printer.h"

namespace tvm {
namespace relay {

Doc TIRTextPrinter::Print(const ObjectRef& node) {
  if (!node.defined()) return Doc::Text("(nullptr)");
  if (node->IsInstance<StmtNode>()) {
    return VisitStmt(Downcast<Stmt>(node));
  } else if (node->IsInstance<AnyNode>()) {
    return Doc::Text("?");
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
  } else if (node->IsInstance<DataProducerNode>()) {
    return PrintProducer(node.as<DataProducerNode>());
  } else if (node->IsInstance<StringObj>()) {
    return PrintString(node.as<StringObj>());
  } else if (node->IsInstance<BufferRegionNode>()) {
    return PrintBufferRegion(node.as<BufferRegionNode>());
  } else if (node->IsInstance<TargetNode>()) {
    return Doc::Text(node.as<TargetNode>()->ToDebugString());
  } else {
    return this->meta_->GetMetaNode(node);
  }
}

Doc TIRTextPrinter::PrintPrimFunc(const PrimFunc& prim_func) {
  const auto* op = prim_func.operator->();
  const auto& signature = op->func_type_annotation();
  // collect Meta in DictAttr
  if (prim_func->attrs.defined()) {
    for (const auto& it : prim_func->attrs->dict) {
      meta_collector_.Collect(it.second);
    }
  }
  // collect buffers in buffer_map
  memo_var_.clear();
  memo_buf_.clear();

  // ordered vars associated with buffers, for consistent printing
  std::vector<tir::Var> buffer_vars_ordered;

  for (tir::Var v : op->params) {
    auto buffer_map_find = op->buffer_map.find(v);
    if (buffer_map_find != op->buffer_map.end()) {
      auto map_data = *buffer_map_find;
      buffer_vars_ordered.push_back(map_data.first);
      memo_buf_[map_data.second] = AllocBuf(map_data.second);
    }
  }

  // print PrimFunc
  Doc doc;
  doc << "primfn"
      << "(";
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
  if (prim_func->attrs.defined()) {
    for (const auto& it : op->attrs->dict) {
      attr_docs.push_back(Doc::StrLiteral(it.first) << ": " << Print(it.second));
    }
    attr_doc << NewLine() << "attr = {" << PrintSep(attr_docs, Doc::Text(", ")) << "}";
    doc << Doc::Indent(2, attr_doc);
  }

  // print all the buffers in the tree
  if (memo_buf_.size() != 0) {
    Doc buffer_doc;
    std::vector<Doc> buffer_docs;
    for (const tir::Var& v : buffer_vars_ordered) {
      const Buffer buf = op->buffer_map[v];
      buffer_docs.push_back(BufferNode2Doc(buf.get(), Print(buf)));
    }
    buffer_doc << NewLine() << "buffers = {";
    buffer_doc << PrintSep(buffer_docs, Doc::Indent(11, Doc::Text(",") << NewLine()));
    doc << Doc::Indent(2, buffer_doc) << "}";
  }

  if (op->buffer_map.size() != 0) {
    // print buffer_map
    std::vector<Doc> buffer_map_doc;
    for (const tir::Var& v : buffer_vars_ordered) {
      const Buffer buf = op->buffer_map[v];
      buffer_map_doc.push_back(Print(v) << ": " << Print(buf));
    }
    doc << Doc::Indent(
        2, NewLine() << "buffer_map = {" << PrintSep(buffer_map_doc, Doc::Text(", ")) << "}");
  }

  doc << PrintBody(op->body);
  return doc;
}

Doc TIRTextPrinter::NewLine() { return Doc::NewLine(); }

Doc TIRTextPrinter::PrintIRModule(const IRModule& module) {
  const auto* op = module.operator->();
  Doc doc;

  Doc body;
  body << NewLine();
  std::vector<Doc> functions;
  for (auto it = op->functions.begin(); it != op->functions.end(); ++it) {
    if ((*it).second.as<PrimFuncNode>()) {
      functions.push_back(Print((*it).second));
    }
  }
  body << TIRTextPrinter::PrintSep(functions, NewLine() << NewLine());
  doc << Doc::Indent(0, body);
  return doc;
}

Doc TIRTextPrinter::PrintArray(const ArrayNode* op) {
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

  if (meta_->InMeta(buffer)) {
    return meta_->GetMetaNode(buffer);
  } else if (memo_buf_.count(buffer)) {
    return memo_buf_[buffer];
  } else {
    memo_buf_[buffer] = AllocBuf(buffer);
    return BufferNode2Doc(op, memo_buf_[buffer]);
  }
}

Doc TIRTextPrinter::PrintProducer(const DataProducerNode* op) {
  const DataProducer& prod = GetRef<DataProducer>(op);

  if (meta_->InMeta(prod)) {
    return meta_->GetMetaNode(prod);
  } else if (memo_producer_.count(prod)) {
    return memo_producer_[prod];
  } else {
    memo_producer_[prod] = AllocProducer(prod);
    return DataProducerNode2Doc(op, memo_producer_[prod]);
  }
}

Doc TIRTextPrinter::BufferNode2Doc(const BufferNode* buf, Doc doc) {
  doc << Doc::Text(": Buffer(") << Print(buf->data) << ", " << PrintDType(buf->dtype) << ", "
      << Print(buf->shape) << ", " << Print(buf->strides);
  if (!is_zero(buf->elem_offset)) {
    doc << ", elem_offset=" << Print(buf->elem_offset);
  }
  if (buf->axis_separators.size()) {
    doc << ", axis_separators=" << Print(buf->axis_separators);
  }
  if (GetRef<Buffer>(buf).scope() != "global") {
    doc << ", scope=" << Doc::StrLiteral(GetRef<Buffer>(buf).scope());
  }
  if (buf->data_alignment != runtime::kAllocAlignment) {
    doc << ", align=" << buf->data_alignment;
  }
  if (buf->offset_factor != 1) {
    doc << ", offset_factor=" << buf->offset_factor;
  }
  if (buf->buffer_type != 1) {
    doc << ", type=" << Doc::StrLiteral("auto");
  }
  return doc << ")";
}

Doc TIRTextPrinter::DataProducerNode2Doc(const DataProducerNode* prod, Doc doc) {
  return doc << Doc::Text(": DataProducer(") << Print(prod->GetNameHint()) << ", "
             << PrintDType(prod->GetDataType()) << ", " << Print(prod->GetShape()) << ")";
}

Doc TIRTextPrinter::PrintBufferRegion(const BufferRegionNode* op) {
  Doc doc;
  doc << Print(op->buffer) << "[";
  for (size_t i = 0; i < op->region.size(); ++i) {
    if (i != 0) {
      doc << ", ";
    }
    const auto& range = op->region[i];
    if (!is_one(range->extent)) {
      doc << Print(range->min) << ":" << Print(range->min + range->extent);
    } else {
      doc << Print(range->min);
    }
  }
  doc << "]";
  return doc;
}

Doc TIRTextPrinter::VisitExprDefault_(const Object* op) {
  return this->meta_->GetMetaNode(GetRef<ObjectRef>(op));
}

Doc TIRTextPrinter::VisitStmtDefault_(const Object* op) {
  return this->meta_->GetMetaNode(GetRef<ObjectRef>(op));
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

Doc TIRTextPrinter::VisitExpr_(const tir::VarNode* op) {
  const tir::Var& var = GetRef<tir::Var>(op);
  return meta_->InMeta(var) ? meta_->GetMetaNode(var) : AllocVar(GetRef<tir::Var>(op));
}

#define TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(OpName, OpString) \
  Doc TIRTextPrinter::VisitExpr_(const OpName* op) {         \
    Doc doc;                                                 \
    doc << "(" << Print(op->a) << OpString;                  \
    doc << Print(op->b) << ")";                              \
    return doc;                                              \
  }

TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(AddNode, " + ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(SubNode, " - ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(MulNode, "*")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(DivNode, " / ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(ModNode, " % ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(EQNode, " == ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(NENode, " != ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(LTNode, " < ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(LENode, " <= ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(GTNode, " > ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(GENode, " >= ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(AndNode, " && ")
TVM_DECLARE_TIR_TEXT_PRINTER_BINOP(OrNode, " || ")

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
      << Print(op->false_value) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const BufferLoadNode* op) {
  Doc doc;
  doc << Print(op->buffer) << Print(op->indices);
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const ProducerLoadNode* op) {
  // TODO(tvm-team): consider make a better text format for producer.
  Doc doc;
  doc << op->producer->GetNameHint() << Print(op->indices);
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

Doc TIRTextPrinter::VisitExpr_(const tir::LetNode* op) {
  Doc doc;
  doc << "let " << Print(op->var) << " = " << Print(op->value) << " in " << Print(op->body);
  return doc;
}

Doc TIRTextPrinter::VisitExpr_(const tir::CallNode* op) {
  Doc doc;
  std::vector<Doc> func_args;
  if (auto* ptr_op = op->op.as<OpNode>()) {
    doc << "@" << Doc::Text(ptr_op->name) << "(";
    if (ptr_op->name == "tir.call_llvm_pure_intrin") {
      auto f = tvm::runtime::Registry::Get("target.llvm_get_intrinsic_name");
      ICHECK(f != nullptr)
          << "Cannot find target.llvm_get_intrinsic_name. Compile with USE_LLVM=On";
      func_args.push_back(Print((*f)(Downcast<IntImm>(op->args[0])->value)));
      for (size_t i = 1; i < op->args.size(); i++) {
        func_args.push_back(Print(op->args[i]));
      }
    } else {
      for (const auto& arg : op->args) {
        func_args.push_back(Print(arg));
      }
    }
  } else {
    // TODO(bohan): Print out the name by he global var in the module.
    auto* op_gvar = op->op.as<GlobalVarNode>();
    ICHECK(op_gvar != nullptr);
    doc << "@" << Doc::Text(op_gvar->name_hint) << "(";
    for (const auto& arg : op->args) {
      func_args.push_back(Print(arg));
    }
  }
  doc << PrintSep(func_args, Doc::Text(", ")) << ", dtype=" << PrintDType(op->dtype) << ")";
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
      << ", " << op->value_index << ", " << Print(op->init) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const LetStmtNode* op) {
  Doc doc;
  doc << "let " << Print(op->var) << " = " << Print(op->value) << NewLine() << Print(op->body);
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
    doc << ";" << NewLine() << Print(op->body);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const AssertStmtNode* op) {
  Doc doc;
  doc << "assert(" << Print(op->condition) << ", " << Print(op->message) << ")" << NewLine()
      << Print(op->body);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const BufferStoreNode* op) {
  Doc doc;
  doc << Print(op->buffer) << Print(op->indices) << " = " << Print(op->value);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const ProducerStoreNode* op) {
  Doc doc;
  doc << Print(op->producer) << Print(op->indices) << " = " << Print(op->value);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const BufferRealizeNode* op) {
  Doc doc;
  doc << "realize(" << Print(op->buffer) << ", " << Print(op->bounds) << ", "
      << Print(op->condition) << PrintBody(op->body) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const ProducerRealizeNode* op) {
  Doc doc;
  doc << "producer_realize(" << Print(op->producer) << ", " << Print(op->bounds) << ", "
      << Print(op->condition) << ", " << PrintBody(op->body) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const AllocateNode* op) {
  Doc doc;
  auto scope = GetPtrStorageScope(op->buffer_var);
  doc << "allocate(" << Print(op->buffer_var) << ", ";
  doc << PrintDType(op->dtype) << ", ";
  doc << Print(op->extents) << "), storage_scope = " << scope;
  if (!op->annotations.empty()) {
    std::vector<Doc> attr_docs;
    for (const auto& it : op->annotations) {
      attr_docs.push_back(Doc::StrLiteral(it.first) << ": " << Print(it.second));
    }
    doc << ", annotations = {" << PrintSep(attr_docs, Doc::Text(", ")) << "})";
  }
  if (!is_one(op->condition)) {
    doc << " if " << Print(op->condition);
  }
  if (op->body->IsInstance<SeqStmtNode>()) {
    doc << PrintBody(op->body);
  } else {
    doc << ";" << NewLine() << Print(op->body);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const AllocateConstNode* op) {
  Doc doc;
  doc << "constant(" << Print(op->buffer_var) << ", " << PrintDType(op->dtype) << ", "
      << Print(op->extents) << ")";

  if (op->body->IsInstance<SeqStmtNode>()) {
    doc << PrintBody(op->body);
  } else {
    doc << ";" << NewLine() << Print(op->body);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const DeclBufferNode* op) {
  Doc doc;
  doc << AllocBuf(op->buffer) << " = decl_buffer(" << Print(op->buffer->data) << ", "
      << PrintDType(op->buffer->dtype) << ", " << Print(op->buffer->shape) << ")" << NewLine();
  if (op->body->IsInstance<SeqStmtNode>()) {
    doc << PrintBody(op->body);
  } else {
    doc << ";" << NewLine() << Print(op->body);
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const IfThenElseNode* op) {
  Doc doc;
  doc << "if " << Print(op->condition) << PrintBody(op->then_case);
  if (!is_one(op->condition) && op->else_case) {
    doc << " else" << PrintBody(op->else_case.value());
  }
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const SeqStmtNode* op) {
  std::vector<Doc> stmts;
  Doc seq_doc, doc;
  for (Stmt stmt : op->seq) {
    seq_doc << NewLine() << Print(stmt);
  }
  doc << " {" << Doc::Indent(2, seq_doc) << NewLine() << "}";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const EvaluateNode* op) {
  Doc doc;
  doc << Print(op->value);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const ForNode* op) {
  Doc doc;
  doc << "for (" << Print(op->loop_var) << ", " << Print(op->min) << ", "
      << Print(op->min + op->extent) << ")";
  if (op->kind != ForKind::kSerial) {
    doc << " " << Doc::StrLiteral(ForKind2String(op->kind));
  }
  doc << PrintBody(op->body);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const WhileNode* op) {
  Doc doc;
  doc << "while (" << Print(op->condition) << ")";
  doc << PrintBody(op->body);
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const PrefetchNode* op) {
  Doc doc;
  doc << "prefetch(" << Print(op->buffer) << ", " << Print(op->bounds) << ")";
  return doc;
}

Doc TIRTextPrinter::VisitStmt_(const BlockRealizeNode* op) {
  const auto* block_op = op->block.as<BlockNode>();
  // print block name and block vars
  Doc doc;
  doc << "block([";
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
          LOG(FATAL) << "Unknown block var iter type";
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
    Doc block_var_name;
    AllocVar(iter_var->var);
    block_var_names.push_back(Print(iter_var->var));
  }
  if (!block_var_names.empty()) {
    doc << " as [" << PrintSep(block_var_names, Doc::Text(", ")) << "]";
  }
  doc << " {";
  Doc block_attr_doc;
  // print predicate, binding, read/write tensor region, annotations
  if (!is_one(op->predicate)) {
    block_attr_doc << NewLine() << "where(" << Print(op->predicate) << ")";
  }
  for (size_t i = 0; i < block_op->iter_vars.size(); ++i)
    block_attr_doc << NewLine() << "bind(" << Print(block_op->iter_vars[i]->var) << ", "
                   << Print(op->iter_values[i]) << ")";
  block_attr_doc << NewLine() << "tir.reads(" << Print(block_op->reads) << ")";
  block_attr_doc << NewLine() << "tir.writes(" << Print(block_op->writes) << ")";
  if (!block_op->annotations.empty()) {
    std::vector<Doc> attr_docs;
    for (const auto& it : block_op->annotations) {
      attr_docs.push_back(Doc::StrLiteral(it.first) << ": " << Print(it.second));
    }
    block_attr_doc << NewLine() << "tir.attrs({" << PrintSep(attr_docs, Doc::Text(", ")) << "})";
  }
  // print body
  Doc body;
  body << NewLine();
  for (const auto& alloc_buf : block_op->alloc_buffers) {
    body << AllocBuf(alloc_buf) << " = alloc_buffer(" << PrintDType(alloc_buf->dtype)
         << Print(alloc_buf->shape) << ")" << NewLine();
  }
  for (const auto& match_buf : block_op->match_buffers) {
    body << AllocBuf(match_buf->buffer) << " = match_buffer(" << Print(match_buf->source) << ")"
         << NewLine();
  }
  if (block_op->init.defined()) {
    Doc init_block;
    init_block << "with init()";
    init_block << PrintBody(block_op->init.value());
    body << init_block << NewLine();
  }
  body << Print(block_op->body);
  doc << Doc::Indent(2, block_attr_doc << body);
  return doc;
}

Doc TIRTextPrinter::VisitType_(const PrimTypeNode* node) {
  Doc doc;
  doc << PrintDType(node->dtype);
  return doc;
}

Doc TIRTextPrinter::VisitType_(const PointerTypeNode* node) {
  Doc doc;
  doc << "Pointer(";
  if (!node->storage_scope.empty()) {
    doc << node->storage_scope << " ";
  }
  doc << Print(node->element_type) << ")";
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

Doc TIRTextPrinter::PrintDType(DataType dtype) {
  return Doc::Text(runtime::DLDataType2String(dtype));
}

template <typename T>
Doc TIRTextPrinter::PrintConstScalar(DataType dtype, const T& data) {
  Doc doc;
  std::ostringstream os;
  os << data;
  if (dtype == DataType::Int(32)) {
    doc << Doc::Text(os.str());
  } else {
    if (dtype.bits() == 1 && dtype.lanes() == 1 && dtype.code() == kDLUInt) {
      doc << ((data == 1) ? "True" : "False");
      return doc;
    }
    doc << Doc::Text(os.str());
    switch (dtype.code()) {
      case kDLInt:
        doc << "i";
        break;
      case kDLUInt:
        doc << "u";
        break;
      case kDLFloat:
        doc << "f";
        break;
    }
    doc << Doc::Text(std::to_string(dtype.bits()));
    if (dtype.lanes() != 1) doc << "x" << Doc::Text(std::to_string(dtype.lanes()));
  }
  return doc;
}

Doc TIRTextPrinter::GetUniqueName(std::string prefix) {
  // std::replace(prefix.begin(), prefix.end(), '.', '_');
  std::string unique_prefix = prefix;
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end()) {
    while (name_alloc_map_.count(unique_prefix = prefix + "_" + std::to_string(++it->second)) > 0) {
    }
  }
  name_alloc_map_[unique_prefix] = 0;
  return Doc::Text(unique_prefix);
}

Doc TIRTextPrinter::AllocVar(const tir::Var& var) {
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
  return val << ": " << Print(GetType(var));
}

Doc TIRTextPrinter::AllocBuf(const Buffer& buffer) {
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

Doc TIRTextPrinter::AllocProducer(const DataProducer& producer) {
  const auto& it = memo_producer_.find(producer);
  if (it != memo_producer_.end()) {
    return it->second;
  }
  std::string name = producer->GetNameHint();
  if (name.length() == 0 || !std::isalpha(name[0])) {
    name = "tensor_" + name;
  }
  Doc val = GetUniqueName(name);
  memo_producer_[producer] = val;
  return val;
}

Doc TIRTextPrinter::PrintSep(const std::vector<Doc>& vec, const Doc& sep) {
  Doc seq;
  if (vec.size() != 0) {
    seq = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
      seq << sep << vec[i];
    }
  }
  return seq;
}

Doc TIRTextPrinter::PrintBody(const Stmt& body, bool indent) {
  Doc doc;
  if (body->IsInstance<SeqStmtNode>()) return Print(body);
  doc << " {" << Doc::Indent(2, NewLine() << Print(body)) << NewLine() << "}";
  return doc;
}

bool TIRTextPrinter::GetVarName(tir::Var v, std::string* s) {
  auto it = memo_var_.find(v);
  if (it == memo_var_.end()) {
    return false;
  }

  *s = it->second.str();
  return true;
}

}  // namespace relay
}  // namespace tvm
