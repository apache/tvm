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

TVM_REGISTER_NODE_TYPE(SourceNode);

/*! \brief Construct a source from a string. */
Source::Source(SourceName src_name, std::string source) {
  auto n = make_object<SourceNode>();
  n->source_name = std::move(src_name);
  n->source = std::move(source);

  int index = 0;
  int length = 0;
  n->line_map.push_back({index, length});
  // NB(@jroesch):
  std::string source_str = n->source;
  for (auto c : source_str) {
    if (c == '\n') {
      // Record the length of the line.
      n->line_map.back().second = length;
      // Bump past the newline.
      index += 1;
      // Record the start of the next line, and put placeholder for length.
      n->line_map.push_back({index, 0});
      // Reset length to zero.
      length = 0;
    } else {
      length += 1;
      index += 1;
    }
  }
  n->line_map.back().second = length;

  data_ = n;
}

tvm::String Source::GetLine(int line) {
  VLOG(1) << "Source::GetLine: line=" << line;
  ICHECK(line - 1 < static_cast<int64_t>((*this)->line_map.size()))
      << "requested line: " << line << "at index: " << (line - 1)
      << "line_map size: " << (*this)->line_map.size() << "source: " << (*this)->source;

  // Adjust for zero indexing, now have (line_start, line_length);
  auto range = (*this)->line_map.at(line - 1);
  int line_start = range.first;
  int line_length = range.second;
  VLOG(1) << "Source::GetLine: line_start=" << line_start << " line_length=" << line_length;
  // TODO(@jroesch): expose substring on tvm::String.
  auto line_text = std::string((*this)->source).substr(line_start, line_length);
  VLOG(1) << "Source::GetLine: line_text=" << line_text;
  return line_text;
}

TVM_REGISTER_NODE_TYPE(SourceMapNode);

SourceMap::SourceMap(Map<SourceName, Source> source_map) {
  auto n = make_object<SourceMapNode>();
  n->source_map = std::move(source_map);
  data_ = std::move(n);
}

void SourceMap::Add(const Source& source) { (*this)->source_map.Set(source->source_name, source); }

TVM_REGISTER_GLOBAL("SourceMapAdd").set_body_typed([](SourceMap map, String name, String content) {
  auto src_name = SourceName::Get(name);
  Source source(src_name, content);
  map.Add(source);
  return src_name;
});

}  // namespace parser
}  // namespace tvm
