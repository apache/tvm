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
 * \file install_debug_spans.cc
 * \brief Prints TIR code in memory and replaces all spans in the module with
    the location to which the ops would be printed
 */

#include "./install_debug_spans.h"

#include <tvm/tir/transform.h>

#include <string>
#include <utility>

#include "../../relay/printer/tir_text_printer_debug.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

Stmt DebugInfoInstaller::InstallInfo(const std::string& name, const Stmt& stmt) {
  DebugInfoInstaller installer(stmt, name + ".tir");
  return installer.VisitStmt(stmt);
}

DebugInfoInstaller::DebugInfoInstaller(const Stmt& stmt, const std::string& filename) {
  // Determine the line that each stmt/expr will be printed on
  tvm::relay::TIRTextPrinterDebug printer(false);

  // Fill in the stmts and exprs' line info
  auto result = printer.Print(stmt).str();

  // Create map of the stmt/expr -> its line number in the output to later
  // create new spans for each stmt/expr
  const auto& stmts = printer.GetStmtsByLine();
  VLOG(0) << "Debug printer found " << stmts.size() << " stmts after printing";
  for (const auto& line : stmts) {
    stmt_lines_[std::get<0>(line)] = std::get<1>(line);
  }

  const auto& exprs = printer.GetExprsByLine();
  VLOG(0) << "Debug printer found " << exprs.size() << " exprs after printing";
  for (const auto& line : exprs) {
    expr_lines_[std::get<0>(line)] = std::get<1>(line);
  }

  // Output the printed TIR to the specified file
  VLOG(0) << "Outputting TIR to " << filename;
  filename_ = std::move(filename);
  std::ofstream out(filename_);
  out << result;
  out.close();
}

PrimExpr DebugInfoInstaller::VisitExpr(const PrimExpr& expr) {
  PrimExpr result = expr;
  result = StmtExprMutator::VisitExpr(result);
  return result;
}

Stmt DebugInfoInstaller::VisitStmt(const Stmt& stmt) {
  Stmt result = stmt;
  result = StmtExprMutator::VisitStmt(result);
  return result;
}

Span DebugInfoInstaller::MaybeSpan(const StmtNode* op) {
  auto entry = stmt_lines_.find(op);
  if (entry == stmt_lines_.end()) {
    return Span();
  } else {
    size_t column = 0;
    size_t line = entry->second;
    return Span(SourceName::Get(filename_), line, line, column, column);
  }
}

Span DebugInfoInstaller::MaybeSpan(const PrimExprNode* op) {
  auto entry = expr_lines_.find(op);
  if (entry == expr_lines_.end()) {
    return Span();
  } else {
    size_t column = 0;
    size_t line = entry->second;
    return Span(SourceName::Get(filename_), line, line, column, column);
  }
}

#define X(TypeName)                                                   \
  PrimExpr DebugInfoInstaller::VisitExpr_(const TypeName##Node* op) { \
    auto new_expr = StmtExprMutator::VisitExpr_(op);                  \
    auto new_type = Downcast<TypeName>(new_expr);                     \
    auto new_node = new_type.CopyOnWrite();                           \
    new_node->span = MaybeSpan(op);                                   \
    return new_type;                                                  \
  }
TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_EXPRS
#undef X

#define X(TypeName)                                               \
  Stmt DebugInfoInstaller::VisitStmt_(const TypeName##Node* op) { \
    Stmt new_stmt = StmtExprMutator::VisitStmt_(op);              \
    auto new_type = Downcast<TypeName>(new_stmt);                 \
    auto new_node = new_type.CopyOnWrite();                       \
    new_node->span = MaybeSpan(op);                               \
    return new_type;                                              \
  }
TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_STMTS
#undef X

namespace transform {

Pass InstallDebugSpans() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    Map<GlobalVar, PrimFunc> external_host_functions;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto prim_func = opt.value();
        if (IsHostFunc(prim_func).value_or(false) &&
            prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
          external_host_functions.Set(gvar, prim_func);
        }
      }
    }

    ICHECK_EQ(external_host_functions.size(), 1)
        << "Debug info can only be added to IRModules with a single host function";

    for (auto [gvar, prim_func] : external_host_functions) {
      auto name = prim_func->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
      prim_func.CopyOnWrite()->body = DebugInfoInstaller::InstallInfo(name, prim_func->body);
      mod.CopyOnWrite()->Update(gvar, prim_func);
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.InstallDebugSpans", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InstallDebugSpans").set_body_typed(InstallDebugSpans);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
