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
 * \file diagnostic.h
 * \brief A new diagnostic interface for TVM error reporting.
 *
 * A prototype of the new diagnostic reporting interface for TVM.
 *
 * Eventually we hope to promote this file to the top-level and
 * replace the existing errors.h.
 */

#ifndef TVM_PARSER_DIAGNOSTIC_H_
#define TVM_PARSER_DIAGNOSTIC_H_

#include <fstream>
#include <tvm/runtime/object.h>
#include <tvm/runtime/container.h>
#include <tvm/ir/span.h>

namespace tvm {
namespace parser {

struct Source {
  std::string source;
  std::vector<int> line_map;
  Source() : source(), line_map() {}
  Source(const std::string& source) : source(source) {
    line_map.push_back(0);
    int i = 0;
    for (auto c : source) {
      i++;
      if (c == '\n') {
        line_map.push_back(i);
      }
    }
    line_map.push_back(i);
  }

  void ReportAt(int line, int column, const std::string& msg) const {
    int line_start = line_map.at(line - 1);
    int line_end = line_map.at(line) - 2;
    int line_length = line_end - line_start;
    std::cout << "file:" << line << ":" << column << ": parse error: " << msg << std::endl;
    std::cout << "    " << source.substr(line_start, line_end) << std::endl;
    std::cout << "    ";
    std::stringstream marker;
    for (int i = 1; i <= line_length; i++) {
      if (i == column) {
        marker << "^";
      } else if ((column - i) < 3) {
        marker << "~";
      } else if ((i - column) < 3) {
        marker << "~";
      } else {
        marker << " ";
      }
    }
    std::cout << marker.str();
    std::cout << std::endl;
  }
};

enum DiagnosticLevel {
  Bug,
  Error,
  Warning,
  Note,
  Help,
};

struct Diagnostic {
  DiagnosticLevel level;
  Span span;
  std::string message;
  Diagnostic(int line, int column, const std::string& message) : level(DiagnosticLevel::Error), span(SourceName(), line, column), message(message) {}
};

struct DiagnosticContext {
  const Source& source;
  std::vector<Diagnostic> diagnostics;

  DiagnosticContext(const Source& source) : source(source) {}

  void Emit(const Diagnostic& diagnostic) {
      diagnostics.push_back(diagnostic);
  }

  void Render() {
      for (auto diagnostic : diagnostics) {
          source.ReportAt(diagnostic.span->line, diagnostic.span->column, diagnostic.message);
      }

      if (diagnostics.size()) {
          exit(1);
      }
  }
};

}  // namespace parser
}  // namespace tvm
#endif  // TVM_PARSER_DIAGNOSTIC_H_
