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

#include "tir_text_printer_debug.h"

#include <string>

#include "text_printer.h"

namespace tvm {
namespace tir {

std::string span_text(const Span& span) {
  if (!span.defined()) {
    return "missing";
  }
  std::string source("file");
  return source + ":" + std::to_string(span->line) + ":" + std::to_string(span->column);
}

Doc TIRTextPrinterDebug::NewLine() {
  current_line_ += 1;

  return TIRTextPrinter::NewLine();
}

#define X(TypeName)                                               \
  Doc TIRTextPrinterDebug::VisitExpr_(const TypeName##Node* op) { \
    exprs_by_line_.push_back(std::make_tuple(op, current_line_)); \
    return TIRTextPrinter::VisitExpr_(op);                        \
  }
TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_EXPRS
#undef X

#define X(TypeName)                                               \
  Doc TIRTextPrinterDebug::VisitStmt_(const TypeName##Node* op) { \
    stmts_by_line_.push_back(std::make_tuple(op, current_line_)); \
    return TIRTextPrinter::VisitStmt_(op);                        \
  }
TVM_TIR_TRANSFORMS_INSTALL_DEBUG_SPANS_SUPPORTED_STMTS
#undef X

}  // namespace tir
}  // namespace tvm
