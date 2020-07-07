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

#include <tvm/ir/span.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace parser {

/*! \brief A program source in any language.
 *
 * Could represent the source from an ML framework or the internal
 * source of a TVM program.
 */
struct Source {
  /*! \brief The raw source. */
  std::string source;
  /*! \brief A mapping of line breaks into the raw source. */
  std::vector<std::pair<int, int>> line_map;

  /*! \brief An empty source. */
  Source() : source(), line_map() {}

  /*! \brief Construct a source from a string. */
  explicit Source(const std::string& source) : source(source) {
    int index = 0;
    int length = 0;
    line_map.push_back({index, length});
    for (auto c : source) {
      if (c == '\n') {
        // Record the length of the line.
        line_map.back().second = length;
        // Bump past the newline.
        index += 1;
        // Record the start of the next line, and put placeholder for length.
        line_map.push_back({index, 0});
        // Reset length to zero.
        length = 0;
      } else {
        length += 1;
        index += 1;
      }
    }
    line_map.back().second = length;
  }

  Source(const Source& source) : source(source.source), line_map(source.line_map) {}

  /*! \brief Generate an error message at a specific line and column with the
   * annotated message.
   *
   * The error is written directly to the `out` std::ostream.
   *
   * \param out The output ostream.
   * \param line The line at which to report a diagnostic.
   * \param line The column at which to report a diagnostic.
   * \param msg The message to attach.
   */
  void ReportAt(std::ostream& out, int line, int column, const std::string& msg) const {
    CHECK(line - 1 <= static_cast<int64_t>(line_map.size()))
        << "requested line: " << (line - 1) << "line_map size: " << line_map.size()
        << "source: " << source;

    // Adjust for zero indexing, now have (line_start, line_length);
    auto range = line_map.at(line - 1);
    int line_start = range.first;
    int line_length = range.second;
    out << "file:" << line << ":" << column << ": parse error: " << msg << std::endl;
    out << "    " << source.substr(line_start, line_length) << std::endl;
    out << "    ";
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
    out << marker.str();
    out << std::endl;
  }
};

/*! \brief The diagnostic level, controls the printing of the message. */
enum DiagnosticLevel {
  Bug,
  Error,
  Warning,
  Note,
  Help,
};

/*! \brief A diagnostic message. */
struct Diagnostic {
  /*! \brief The level. */
  DiagnosticLevel level;
  /*! \brief The span at which to report an error. */
  Span span;
  /*! \brief The diagnostic message. */
  std::string message;

  Diagnostic(int line, int column, const std::string& message)
      : level(DiagnosticLevel::Error), span(SourceName(), line, column), message(message) {}
};

/*! \brief A diagnostic context for recording errors against a source file.
 * TODO(@jroesch): convert source map and improve in follow up PR, the parser
 * assumes a single global file for now.
 */
struct DiagnosticContext {
  /*! \brief The source to report against. */
  Source source;

  /*! \brief The set of diagnostics to report. */
  std::vector<Diagnostic> diagnostics;

  explicit DiagnosticContext(const Source& source) : source(source) {}

  /*! \brief Emit a diagnostic. */
  void Emit(const Diagnostic& diagnostic) { diagnostics.push_back(diagnostic); }

  // TODO(@jroesch): eventually modularize the rendering interface to provide control of how to
  // format errors.
  void Render(std::ostream& ostream) {
    for (auto diagnostic : diagnostics) {
      source.ReportAt(ostream, diagnostic.span->line, diagnostic.span->column, diagnostic.message);
    }

    if (diagnostics.size()) {
      LOG(FATAL) << "parse error occured";
    }
  }
};

}  // namespace parser
}  // namespace tvm
#endif  // TVM_PARSER_DIAGNOSTIC_H_
