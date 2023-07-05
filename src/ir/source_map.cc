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
#include <tvm/ir/source_map.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>

#include <algorithm>

namespace tvm {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.frontend.fill_span", Bool);

ObjectPtr<Object> GetSourceNameNode(const String& name) {
  // always return pointer as the reference can change as map re-allocate.
  // or use another level of indirection by creating a unique_ptr
  static std::unordered_map<String, ObjectPtr<SourceNameNode>> source_map;

  auto sn = source_map.find(name);
  if (sn == source_map.end()) {
    ObjectPtr<SourceNameNode> n = make_object<SourceNameNode>();
    source_map[name] = n;
    n->name = std::move(name);
    return n;
  } else {
    return sn->second;
  }
}

ObjectPtr<Object> GetSourceNameNodeByStr(const std::string& name) {
  return GetSourceNameNode(name);
}

SourceName SourceName::Get(const String& name) { return SourceName(GetSourceNameNode(name)); }

TVM_REGISTER_GLOBAL("ir.SourceName").set_body_typed(SourceName::Get);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SourceNameNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const SourceNameNode*>(ref.get());
      p->stream << "SourceName(" << node->name << ", " << node << ")";
    });

TVM_REGISTER_NODE_TYPE(SourceNameNode)
    .set_creator(GetSourceNameNodeByStr)
    .set_repr_bytes([](const Object* n) -> std::string {
      return static_cast<const SourceNameNode*>(n)->name;
    });

Span::Span(SourceName source_name, int line, int end_line, int column, int end_column) {
  auto n = make_object<SpanNode>();
  n->source_name = std::move(source_name);
  n->line = line;
  n->end_line = end_line;
  n->column = column;
  n->end_column = end_column;
  data_ = std::move(n);
}

Span Span::Merge(const Span& other) const {
  ICHECK(this->defined() && other.defined()) << "Span::Merge: both spans must be defined";

  ICHECK((*this)->source_name == other->source_name);
  return Span((*this)->source_name, std::min((*this)->line, other->line),
              std::max((*this)->end_line, other->end_line),
              std::min((*this)->column, other->column),
              std::max((*this)->end_column, other->end_column));
}

TVM_REGISTER_NODE_TYPE(SpanNode);

SequentialSpan::SequentialSpan(tvm::Array<Span> spans) {
  auto n = make_object<SequentialSpanNode>();
  tvm::Array<Span> tmp_spans;
  for (const Span& s : spans) {
    if (const SequentialSpanNode* seq_s = s.as<SequentialSpanNode>()) {
      tmp_spans.insert(tmp_spans.end(), seq_s->spans.begin(), seq_s->spans.end());
    } else {
      tmp_spans.push_back(s);
    }
  }
  n->spans = std::move(tmp_spans);

  n->line = 0;
  n->end_line = 0;
  n->column = 0;
  n->end_column = 0;

  data_ = std::move(n);
}

SequentialSpan::SequentialSpan(std::initializer_list<Span> init) {
  auto n = make_object<SequentialSpanNode>();
  tvm::Array<Span> spans = tvm::Array<Span>(init);
  tvm::Array<Span> tmp_spans;
  for (const Span& s : spans) {
    if (const SequentialSpanNode* seq_s = s.as<SequentialSpanNode>()) {
      tmp_spans.insert(tmp_spans.end(), seq_s->spans.begin(), seq_s->spans.end());
    } else {
      tmp_spans.push_back(s);
    }
  }
  n->spans = std::move(tmp_spans);

  n->line = 0;
  n->end_line = 0;
  n->column = 0;
  n->end_column = 0;

  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(SequentialSpanNode);

TVM_REGISTER_GLOBAL("ir.Span").set_body_typed([](SourceName source_name, int line, int end_line,
                                                 int column, int end_column) {
  return Span(source_name, line, end_line, column, end_column);
});

TVM_REGISTER_GLOBAL("ir.SequentialSpan").set_body_typed([](tvm::Array<Span> spans) {
  return SequentialSpan(spans);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SpanNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const SpanNode*>(ref.get());
      p->stream << "Span(" << node->source_name << ", " << node->line << ", " << node->end_line
                << ", " << node->column << ", " << node->end_column << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SequentialSpanNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const SequentialSpanNode*>(ref.get());

      p->stream << "SequentailSpan([ ";
      int index = 0;
      const int last = node->spans.size() - 1;
      while (index < last) {
        p->stream << node->spans[index++] << ", ";
      }
      p->stream << node->spans[last] << " ])";
    });

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

}  // namespace tvm
