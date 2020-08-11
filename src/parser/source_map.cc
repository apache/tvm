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
 * \file source_map.cc
 * \brief The implementation of the source map data structure.
 */
#include <tvm/parser/source_map.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace parser {

/*! \brief Construct a source from a string. */
Source::Source(const SourceName& src_name, const std::string& source)
    : source_name(src_name), source(source) {
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
void Source::ReportAt(std::ostream& out, const Span& span, const std::string& msg) const {
  DLOG(INFO) << "Source::ReportAt"
             << "span = " << span << "msg = " << msg;
  int line = span->line;
  int column = span->column;

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

// TVM_REGISTER_GLOBAL("ir.SourceName").set_body_typed(SourceName::Get);

// TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
//     .set_dispatch<SourceNameNode>([](const ObjectRef& ref, ReprPrinter* p) {
//       auto* node = static_cast<const SourceNameNode*>(ref.get());
//       p->stream << "SourceName(" << node->name << ", " << node << ")";
//     });

TVM_REGISTER_NODE_TYPE(SourceMapNode);

SourceMap::SourceMap(Map<SourceName, tvm::String> source_map) {
  auto n = make_object<SourceMapNode>();
  n->source_map = std::move(source_map);
  data_ = std::move(n);
}

}  // namespace parser
}  // namespace tvm
