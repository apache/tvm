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
 * \file source_map.h
 * \brief A map from source names to source code.
 */
#ifndef TVM_IR_SOURCE_MAP_H_
#define TVM_IR_SOURCE_MAP_H_

#include <tvm/node/node.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief The source name in the Span
 * \sa SourceNameNode, Span
 */
class SourceName;
/*!
 * \brief The name of a source fragment.
 */
class SourceNameNode : public Object {
 public:
  /*! \brief The source name. */
  String name;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  static constexpr bool _type_has_method_sequal_reduce = true;

  bool SEqualReduce(const SourceNameNode* other, SEqualReducer equal) const {
    return equal(name, other->name);
  }

  static constexpr const char* _type_key = "SourceName";
  TVM_DECLARE_FINAL_OBJECT_INFO(SourceNameNode, Object);
};

/*!
 * \brief The source name of a file span.
 * \sa SourceNameNode, Span
 */
class SourceName : public ObjectRef {
 public:
  /*!
   * \brief Get an SourceName for a given operator name.
   *  Will raise an error if the source name has not been registered.
   * \param name Name of the operator.
   * \return SourceName valid throughout program lifetime.
   */
  TVM_DLL static SourceName Get(const String& name);

  TVM_DEFINE_OBJECT_REF_METHODS(SourceName, ObjectRef, SourceNameNode);
};

/*!
 * \brief Span information for debugging purposes
 */
class Span;
/*!
 * \brief Stores locations in frontend source that generated a node.
 */
class SpanNode : public Object {
 public:
  /*! \brief The source name. */
  SourceName source_name;
  /*! \brief The line number. */
  int line;
  /*! \brief The column offset. */
  int column;
  /*! \brief The end line number. */
  int end_line;
  /*! \brief The end column number. */
  int end_column;

  // override attr visitor
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("source_name", &source_name);
    v->Visit("line", &line);
    v->Visit("column", &column);
    v->Visit("end_line", &end_line);
    v->Visit("end_column", &end_column);
  }
  static constexpr bool _type_has_method_sequal_reduce = true;

  bool SEqualReduce(const SpanNode* other, SEqualReducer equal) const {
    return equal(source_name, other->source_name) && equal(line, other->line) &&
           equal(column, other->column) && equal(end_line, other->end_line) &&
           equal(end_column, other->end_column);
  }

  static constexpr const char* _type_key = "Span";
  TVM_DECLARE_BASE_OBJECT_INFO(SpanNode, Object);
};

class Span : public ObjectRef {
 public:
  TVM_DLL Span(SourceName source_name, int line, int end_line, int column, int end_column);

  /*! \brief Merge two spans into one which captures the combined regions. */
  TVM_DLL Span Merge(const Span& other) const;

  TVM_DEFINE_OBJECT_REF_METHODS(Span, ObjectRef, SpanNode);
};

/*!
 * \brief Store a list of spans for an expr generated from mulitple source exprs
 */
class SequentialSpanNode : public SpanNode {
 public:
  /*! \brief The original source list of spans to construct a sequential span. */
  Array<Span> spans;

  // override attr visitor
  void VisitAttrs(AttrVisitor* v) {
    SpanNode::VisitAttrs(v);
    v->Visit("spans", &spans);
  }

  static constexpr const char* _type_key = "SequentialSpan";
  TVM_DECLARE_FINAL_OBJECT_INFO(SequentialSpanNode, SpanNode);

  bool SEqualReduce(const SequentialSpanNode* other, SEqualReducer equal) const {
    if (spans.size() != other->spans.size()) {
      return false;
    }

    for (size_t i = 0, e = spans.size(); i != e; ++i) {
      if (!StructuralEqual()(spans[i], other->spans[i])) {
        return false;
      }
    }
    return true;
  }
};

/*!
 * \brief Reference class of SequentialSpanNode.
 * \sa SequentialSpanNode
 */
class SequentialSpan : public Span {
 public:
  TVM_DLL SequentialSpan(Array<Span> spans);

  TVM_DLL SequentialSpan(std::initializer_list<Span> init);

  TVM_DEFINE_OBJECT_REF_METHODS(SequentialSpan, Span, SequentialSpanNode);
};

/*! \brief A program source in any language.
 *
 * Could represent the source from an ML framework or a source
 * representing a tvm::IRModule.
 */
class Source;

class SourceNode : public Object {
 public:
  /*! \brief The source name. */
  SourceName source_name;

  /*! \brief The raw source. */
  String source;

  /*! \brief A mapping of line breaks into the raw source. */
  std::vector<std::pair<int, int>> line_map;

  // override attr visitor
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("source_name", &source_name);
    v->Visit("source", &source);
  }

  static constexpr const char* _type_key = "Source";
  TVM_DECLARE_FINAL_OBJECT_INFO(SourceNode, Object);
};

class Source : public ObjectRef {
 public:
  TVM_DLL Source(SourceName src_name, std::string source);
  TVM_DLL tvm::String GetLine(int line);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Source, ObjectRef, SourceNode);
};

/*!
 * \brief A mapping from a unique source name to source fragment.
 */
class SourceMap;
/*!
 * \brief Stores locations in frontend source that generated a node.
 */
class SourceMapNode : public Object {
 public:
  /*! \brief The source mapping. */
  Map<SourceName, Source> source_map;

  // override attr visitor
  void VisitAttrs(AttrVisitor* v) { v->Visit("source_map", &source_map); }

  bool SEqualReduce(const SourceMapNode* other, SEqualReducer equal) const {
    return equal(source_map, other->source_map);
  }

  static constexpr const char* _type_key = "SourceMap";
  TVM_DECLARE_FINAL_OBJECT_INFO(SourceMapNode, Object);
};

class SourceMap : public ObjectRef {
 public:
  explicit SourceMap(Map<SourceName, Source> source_map);

  explicit SourceMap(std::initializer_list<std::pair<SourceName, Source>> source_map)
      : SourceMap(Map<SourceName, Source>(source_map)) {}

  SourceMap() : SourceMap(Map<SourceName, Source>()) {}

  void Add(const Source& source);

  SourceMapNode* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<SourceMapNode*>(get_mutable());
  }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SourceMap, ObjectRef, SourceMapNode);
};

}  // namespace tvm

#endif  // TVM_IR_SOURCE_MAP_H_
