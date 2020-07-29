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
#include <tvm/parser/source_map.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>


#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace parser {

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

  /*! \brief A diagnostic for a single character token. */
  Diagnostic(int line, int column, const std::string& message)
      : level(DiagnosticLevel::Error), span(SourceName(), line, column, line, column + 1), message(message) {}

  Diagnostic(DiagnosticLevel level, Span span, const std::string& message)
    : level(level), span(span), message(message) {}
};

/*!
 * \brief A wrapper around std::stringstream to build a diagnostic.
 *
 * \code
 *
 * void ReportError(const Error& err);
 *
 * void Test(int number) {
 *   // Use error reporter to construct an error.
 *   ReportError(ErrorBuilder() << "This is an error number=" << number);
 * }
 *
 * \endcode
 */
struct DiagnosticBuilder {
 public:
  /*! \brief The level. */
  DiagnosticLevel level;

  /*! \brief The source name. */
  SourceName source_name;

  /*! \brief The line number. */
  int line;

  /*! \brief The column number. */
  int column;

   /*! \brief The line number. */
  int end_line;

  /*! \brief The column number. */
  int end_column;

  template <typename T>
  DiagnosticBuilder& operator<<(const T& val) {  // NOLINT(*)
    stream_ << val;
    return *this;
  }

  DiagnosticBuilder() : level(DiagnosticLevel::Error), source_name(), line(0), column(0) {}
  DiagnosticBuilder(const DiagnosticBuilder& builder)
    : level(builder.level), source_name(builder.source_name), line(builder.line), column(builder.column) {}
  DiagnosticBuilder(DiagnosticLevel level, SourceName source_name, int line, int column)
    : level(level), source_name(source_name), line(line), column(column) {}

  operator Diagnostic() {
    auto span = Span(this->source_name, this->line, this->column, this->end_line, this->end_column);
    return Diagnostic(this->level, span, this->stream_.str());
  }

 private:
  std::stringstream stream_;
  friend struct Diagnostic;
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

  /*! \brief Emit a diagnostic. */
  void EmitFatal(const Diagnostic& diagnostic) {
    diagnostics.push_back(diagnostic);
    Render(std::cout);
    // TODO(@jroesch): throw exception which is caught at the pass boundary and then rendered.
    LOG(FATAL) << "error occurred";
  }

  // TODO(@jroesch): eventually modularize the rendering interface to provide control of how to
  // format errors.
  void Render(std::ostream& ostream) {
    for (auto diagnostic : diagnostics) {
      source.ReportAt(ostream, diagnostic.span->line, diagnostic.span->column, diagnostic.message);
    }

    if (diagnostics.size()) {
      LOG(FATAL) << "parse error occurred";
    }
  }
};

}  // namespace parser
}  // namespace tvm
#endif  // TVM_PARSER_DIAGNOSTIC_H_
