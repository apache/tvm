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
#ifndef TVM_TIRX_IR_SCRIPT_PRINT_UTILS_H_
#define TVM_TIRX_IR_SCRIPT_PRINT_UTILS_H_

#include <tvm/runtime/device_api.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>

#include "../../ir/printer_utils.h"

namespace tvm {
namespace printer {

using namespace tirx;

/*! \brief Define a TIR variable in the printer and return its IdAST. */
inline text::ExprAST DefineVar(const tirx::Var& var, const text::IRPrinter& printer,
                         const text::AccessPath& path) {
  if (printer->VarGet(var).has_value()) {
    return printer->VarGet(var).value();
  }
  text::DefaultFrame frame = printer->frames.back().cast<text::DefaultFrame>();
  printer->VarDef(var->name_hint, var, frame);
  return printer->VarGet(var).value();
}

/*! \brief Define a TIR variable and return AssignAST with annotation if typed. */
inline text::StmtAST DefineVarAssign(const tirx::Var& var, text::ExprAST rhs,
                                const text::IRPrinter& printer, const text::AccessPath& path) {
  text::ExprAST id = DefineVar(var, printer, path);
  ffi::Optional<text::ExprAST> annotation;
  if (var->type_annotation.defined()) {
    annotation = Print(printer, var->type_annotation, path->Attr("type_annotation"));
  }
  return text::AssignAST(id, rhs, annotation);
}

/*! \brief Print a statement body, flattening SeqStmt. Returns a List<StmtAST>. */
inline ffi::List<text::StmtAST> PrintBodyStmts(const Stmt& stmt, const text::IRPrinter& printer,
                                     const text::AccessPath& path) {
  ffi::List<text::StmtAST> result;
  text::NodeAST ast = printer->operator()(ffi::Any(stmt), path).cast<text::NodeAST>();
  if (auto* block = ast.as<text::StmtBlockASTObj>()) {
    for (const auto& s : block->stmts) {
      result.push_back(s);
    }
  } else if (ast->IsInstance<text::StmtASTObj>()) {
    result.push_back(Downcast<text::StmtAST>(ast));
  } else if (ast->IsInstance<text::ExprASTObj>()) {
    result.push_back(text::ExprStmtAST(Downcast<text::ExprAST>(ast)));
  }
  return result;
}

/*! \brief A Var occurrence counter visitor (matches V1 OccurrenceCounter). */
class OccurrenceCounter : public tirx::StmtExprVisitor {
 public:
  int count = 0;
  const tirx::VarNode* v = nullptr;

  void VisitExpr_(const tirx::VarNode* op) final {
    if (op == v) ++count;
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const tirx::BufferStoreNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const tirx::BufferLoadNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const tirx::AllocBufferNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const tirx::DeclBufferNode* op) final {
    VisitBuffer(op->buffer.get());
    tirx::StmtExprVisitor::VisitStmt_(op);
  }
  void VisitBuffer(const tirx::BufferNode* buffer) {
    VisitExpr(buffer->data);
    for (const PrimExpr& shape_i : buffer->shape) VisitExpr(shape_i);
    for (const PrimExpr& stride_i : buffer->strides) VisitExpr(stride_i);
    VisitExpr(buffer->elem_offset);
  }
  explicit OccurrenceCounter(const tirx::VarNode* var) { v = var; }
};

/*! \brief Count how many times a Var occurs in a PrimFunc (params + buffer_map + body). */
inline int CountVarOccurrence(const tirx::PrimFunc& f, const tirx::Var& v) {
  OccurrenceCounter counter(v.get());
  counter(f->body);
  for (const tirx::Var& param : f->params) {
    counter(param);
  }
  for (const auto& pair : f->buffer_map) {
    counter(pair.first);
    counter.VisitBuffer(pair.second.get());
  }
  return counter.count;
}

/*! \brief Check if a buffer is "simple" (can be inlined as param annotation).
 *  Matches V1 IsSimpleBuffer logic exactly. */
inline bool IsSimpleBuffer(const tirx::Buffer& buf) {
  if (!buf->strides.empty()) return false;
  for (const PrimExpr& shp_i : buf->shape) {
    if (!tirx::UndefinedVars(shp_i).empty()) return false;
  }
  if (!tirx::UndefinedVars(buf->elem_offset).empty()) {
    return false;
  } else if (buf->elem_offset->IsInstance<IntImmNode>()) {
    IntImm elem_offset = Downcast<IntImm>(buf->elem_offset);
    if (elem_offset->value != 0) return false;
  }
  return buf.scope() == "global" &&
         buf->data_alignment == runtime::kAllocAlignment &&
         buf->offset_factor == 1 &&
         buf->buffer_type == tirx::BufferType::kDefault &&
         buf->axis_separators.empty();
}

/*! \brief Print buffer as T.Buffer(shape, dtype) annotation (simple buffer). */
inline text::ExprAST PrintBufferAnnotation(const tirx::Buffer& buf, const text::IRPrinter& printer,
                                      const text::AccessPath& path) {
  ffi::List<text::ExprAST> args;
  args.push_back(printer->PrintTuple(buf->shape, path->Attr("shape")));
  args.push_back(text::LiteralAST::Str(DType2Str(buf->dtype)));
  return text::ExprCall(TIR("Buffer"), std::move(args));
}

/*!
 * \brief Check if a PrimExpr is a new (undefined) Var in the printer.
 */
inline bool IsNewVar(const PrimExpr& e, const text::IRPrinter& printer) {
  return e->IsInstance<tirx::VarNode>() && !printer->VarGet(e).has_value();
}

/*!
 * \brief Define a new TIR Var if not already defined, and emit
 *        `var_name = T.<dtype>()` into the frame.
 *        Returns the IdAST for the newly defined var.
 */
inline text::ExprAST DefineNewTIRVar(const tirx::Var& var, const text::IRPrinter& printer,
                                text::DefaultFrame& frame) {
  text::ExprAST var_id = DefineVar(var, printer, text::AccessPath::Root());
  std::string dtype_str = DType2Str(var->dtype);
  // Match V1's PrintVarCreation: add is_size_var=True kwarg for SizeVar
  if (var->IsInstance<tirx::SizeVarNode>()) {
    text::ExprAST rhs = text::ExprCallKw(TIR(dtype_str), {},
                              {ffi::String("is_size_var")}, {text::LiteralAST::Bool(true)});
    frame->stmts.push_back(text::AssignAST(var_id, rhs, ffi::Optional<text::ExprAST>()));
  } else {
    text::ExprAST rhs = text::ExprCall(TIR(dtype_str), {});
    frame->stmts.push_back(text::AssignAST(var_id, rhs, ffi::Optional<text::ExprAST>()));
  }
  return var_id;
}

/*!
 * \brief Define any new vars in a buffer's shape, strides, and elem_offset.
 *        Emits `var = T.int32()` etc. into the frame for each undefined var.
 *        Must be called BEFORE PrintBufferDecl so that all buffer vars are available.
 *
 *        Uses PostOrderVisit to recurse into compound expressions (e.g. batch_size + 1)
 *        to find ALL nested Vars, not just top-level ones.
 */
inline void DefineBufferVars(const tirx::Buffer& buf, const text::IRPrinter& printer,
                              text::DefaultFrame& frame) {
  auto visit_expr = [&](const PrimExpr& e) {
    tirx::PostOrderVisit(e, [&](const ffi::ObjectRef& obj) {
      if (const auto* var_node = obj.as<tirx::VarNode>()) {
        tirx::Var var = ffi::GetRef<tirx::Var>(var_node);
        if (!printer->VarGet(var).has_value()) {
          DefineNewTIRVar(var, printer, frame);
        }
      }
    });
  };
  for (const PrimExpr& e : buf->shape) {
    visit_expr(e);
  }
  for (const PrimExpr& e : buf->strides) {
    visit_expr(e);
  }
  // NOTE: Do NOT define elem_offset vars here. They are handled by
  // PrintBufferDecl which decides whether to emit elem_offset=...
  // kwarg or offset_factor=... with inline definition (matching V1
  // try_inline_def logic).
}

/*!
 * \brief Define the buffer's data variable as `buf_name.data` using VarDefNoName.
 *        This allows references to buf->data to render as `A.data` instead of `A_1`.
 *        Must be called AFTER the buffer itself has been defined via VarDef.
 */
inline void DefineBufferDataVar(const tirx::Buffer& buf, const text::IRPrinter& printer) {
  if (!printer->VarGet(buf->data).has_value()) {
    text::ExprAST buf_expr = printer->VarGet(buf).value();
    // Capture buf_expr in a Function that returns buf_name.data
    ffi::Function creator = ffi::Function::FromTyped([buf_expr]() -> text::ExprAST {
      return text::ExprAttr(buf_expr, "data");
    });
    printer->VarDefNoName(creator, buf->data,
                          ffi::Optional<ffi::ObjectRef>(printer->frames.back().cast<ffi::ObjectRef>()));
  }
}

/*!
 * \brief Print a buffer declaration call: T.<method>(extra_args..., shape, dtype, kwargs...)
 *
 * Matches V1's BufferDecl/BufferCall: positional args are (extra_args..., shape, dtype),
 * then kwargs for non-default: data, strides, elem_offset, scope, align, offset_factor,
 * buffer_type, axis_separators, annotations.
 *
 * NOTE: Call DefineBufferVars() first to define any new shape/stride vars.
 *
 * \param annotations Optional annotations map (from AllocBuffer). If non-empty,
 *        emits annotations={...} kwarg.
 * \param annotations_path AccessPath for annotations (used only when annotations non-empty).
 */
inline text::ExprAST PrintBufferDecl(const tirx::Buffer& buf, const std::string& method,
                                ffi::List<text::ExprAST> extra_args,
                                const text::IRPrinter& printer, const text::AccessPath& path,
                                const ffi::Map<ffi::String, ffi::Any>& annotations = {},
                                const text::AccessPath& annotations_path = text::AccessPath::Root()) {
  // Positional: extra_args, shape, dtype
  ffi::List<text::ExprAST> args;
  for (const auto& a : extra_args) args.push_back(a);
  args.push_back(printer->PrintTuple(buf->shape, path->Attr("shape")));
  if (DType2Str(buf->dtype) != "float32") {
    args.push_back(text::LiteralAST::Str(DType2Str(buf->dtype)));
  }
  // Kwargs for non-default fields
  ffi::List<ffi::String> kw_keys;
  ffi::List<text::ExprAST> kw_vals;
  // data: print for decl_buffer/match_buffer when the data pointer is shared with
  // another already-defined buffer. Skip for alloc_buffer/sblock_alloc_buffer
  // (they create their own data pointer, so data= would be self-referential).
  if (method != "alloc_buffer" && method != "sblock_alloc_buffer" &&
      !IsNewVar(buf->data, printer)) {
    kw_keys.push_back(ffi::String("data"));
    kw_vals.push_back(Print(printer, buf->data, path->Attr("data")));
  }
  // strides (skip for alloc_buffer — its parser doesn't accept strides;
  // sblock_alloc_buffer does accept strides, so emit them)
  if (!buf->strides.empty() && method != "alloc_buffer") {
    kw_keys.push_back(ffi::String("strides"));
    ffi::List<text::ExprAST> stride_elts;
    for (int i = 0; i < static_cast<int>(buf->strides.size()); ++i) {
      stride_elts.push_back(Print(printer, buf->strides[i], path->Attr("strides")->ArrayItem(i)));
    }
    kw_vals.push_back(text::TupleAST({}, std::move(stride_elts)));
  }
  // elem_offset
  // V1 logic: if IntImm, print only if non-zero.
  // If new var, DON'T print elem_offset kwarg (but set needs_print_factor).
  // If existing var, print elem_offset kwarg.
  bool needs_print_factor_for_elem_offset = false;
  if (const auto* int_imm = buf->elem_offset.as<IntImmNode>()) {
    if (int_imm->value != 0) {
      kw_keys.push_back(ffi::String("elem_offset"));
      kw_vals.push_back(Print(printer, buf->elem_offset, path->Attr("elem_offset")));
    }
  } else if (IsNewVar(buf->elem_offset, printer)) {
    // New var: don't print elem_offset kwarg, but force offset_factor printing.
    // Inline-define the var as buf.elem_offset (matching V1's try_inline_def)
    // so subsequent references resolve correctly.
    needs_print_factor_for_elem_offset = true;
    {
      tirx::Var offset_var = Downcast<tirx::Var>(buf->elem_offset);
      text::ExprAST buf_expr = printer->VarGet(buf).value();
      ffi::Function creator = ffi::Function::FromTyped([buf_expr]() -> text::ExprAST {
        return text::ExprAttr(buf_expr, "elem_offset");
      });
      printer->VarDefNoName(creator, offset_var,
                            ffi::Optional<ffi::ObjectRef>(printer->frames.back().cast<ffi::ObjectRef>()));
    }
  } else {
    // Existing var: print elem_offset kwarg
    kw_keys.push_back(ffi::String("elem_offset"));
    kw_vals.push_back(Print(printer, buf->elem_offset, path->Attr("elem_offset")));
  }
  // scope
  if (buf.scope() != "global") {
    kw_keys.push_back(ffi::String("scope"));
    kw_vals.push_back(text::LiteralAST::Str(buf.scope()));
  }
  // align (data_alignment)
  if (buf->data_alignment != runtime::kAllocAlignment) {
    kw_keys.push_back(ffi::String("align"));
    kw_vals.push_back(text::LiteralAST::Int(buf->data_alignment));
  }
  // offset_factor
  if (needs_print_factor_for_elem_offset || buf->offset_factor != 1) {
    kw_keys.push_back(ffi::String("offset_factor"));
    kw_vals.push_back(text::LiteralAST::Int(buf->offset_factor));
  }
  // buffer_type
  if (buf->buffer_type != tirx::BufferType::kDefault) {
    kw_keys.push_back(ffi::String("buffer_type"));
    kw_vals.push_back(text::LiteralAST::Str("auto"));
  }
  // axis_separators (V1 prints for all buffer types except standalone alloc_buffer)
  if (!buf->axis_separators.empty()) {
    kw_keys.push_back(ffi::String("axis_separators"));
    kw_vals.push_back(printer->PrintList(buf->axis_separators, path->Attr("axis_separators")));
  }
  // annotations (from AllocBuffer node, not the Buffer itself)
  if (!annotations.empty()) {
    kw_keys.push_back(ffi::String("annotations"));
    kw_vals.push_back(Print(printer, annotations, annotations_path));
  }

  if (!kw_keys.empty()) {
    return text::ExprCallKw(TIR(method), std::move(args), std::move(kw_keys), std::move(kw_vals));
  }
  return text::ExprCall(TIR(method), std::move(args));
}

}  // namespace printer
}  // namespace tvm

#endif  // TVM_TIRX_IR_SCRIPT_PRINT_UTILS_H_
