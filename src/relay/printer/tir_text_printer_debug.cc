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

#include <optional>
#include <string>

namespace tvm {
namespace relay {

std::optional<std::string> span_text(const Span& span) {
  if (!span.defined()) {
    return std::nullopt;
  }

  std::string source("main.tir");
  if (span->source_name.defined() && span->source_name->name.get()) {
    source = span->source_name->name;
  }
  return source + ":" + std::to_string(span->line) + ":" + std::to_string(span->column);
}

template <typename ObjectPtr>
void add_all_relevant_lines(const std::vector<std::tuple<const ObjectPtr*, size_t>>& data,
                            size_t current_line, Doc* output) {
  ICHECK(output) << "output must be a valid Doc";
  for (const auto& item : data) {
    if (std::get<1>(item) != current_line - 1) {
      // Item is not relevant for this line, skip it
      continue;
    }

    // Print out the item's span info if present
    auto text = span_text(std::get<0>(item)->span);
    if (text.has_value()) {
      *output << *text;
    } else {
      *output << "missing";
    }
    *output << ", ";
  }
}

Doc TIRTextPrinterDebug::NewLine() {
  current_line_ += 1;

  if (!show_spans_) {
    return TIRTextPrinter::NewLine();
  }

  Doc output;

  output << " [";

  add_all_relevant_lines(exprs_by_line_, current_line_, &output);
  add_all_relevant_lines(stmts_by_line_, current_line_, &output);

  output << "]" << TIRTextPrinter::NewLine();

  return output;
}

Doc TIRTextPrinterDebug::VisitStmt(const tvm::tir::Stmt& n) {
  stmts_by_line_.push_back(std::make_tuple(n.get(), current_line_));
  return TIRTextPrinter::VisitStmt(n);
}

Doc TIRTextPrinterDebug::VisitExpr(const PrimExpr& e) {
  exprs_by_line_.push_back(std::make_tuple(e.get(), current_line_));
  return TIRTextPrinter::VisitExpr(e);
}

}  // namespace relay
}  // namespace tvm
