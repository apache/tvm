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
 * \file text_printer.h
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */

#ifndef TVM_RELAY_PRINTER_TIR_TEXT_PRINTER_DEBUG_H_
#define TVM_RELAY_PRINTER_TIR_TEXT_PRINTER_DEBUG_H_

#include <tuple>
#include <vector>

#include "text_printer.h"

namespace tvm {
namespace relay {

class TIRTextPrinterDebug : public TIRTextPrinter {
 public:
  explicit TIRTextPrinterDebug(bool show_spans)
      : TIRTextPrinter(false, &meta_), current_line_(1), show_spans_(show_spans) {}

  std::vector<std::tuple<const PrimExprNode*, size_t>> GetExprsByLine() const {
    return exprs_by_line_;
  }

  std::vector<std::tuple<const StmtNode*, size_t>> GetStmtsByLine() const { return stmts_by_line_; }

 private:
  Doc NewLine() override;

  Doc VisitStmt(const tvm::tir::Stmt& n) override;
  Doc VisitExpr(const PrimExpr& e) override;

  TextMetaDataContext meta_;

  // Line that the printer is currently printing
  size_t current_line_;

  // Whether to include spans relevant to each line before a newline or not
  bool show_spans_;

  // Record of all stmts and exprs and their corresponding line
  std::vector<std::tuple<const StmtNode*, size_t>> stmts_by_line_;
  std::vector<std::tuple<const PrimExprNode*, size_t>> exprs_by_line_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PRINTER_TIR_TEXT_PRINTER_DEBUG_H_
