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
#ifndef TVM_PARSER_SOURCE_MAP_H_
#define TVM_PARSER_SOURCE_MAP_H_

#include <tvm/ir/span.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace parser {

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
  TVM_DLL SourceMap(Map<SourceName, Source> source_map);

  TVM_DLL SourceMap(std::initializer_list<std::pair<SourceName, Source>> source_map)
      : SourceMap(Map<SourceName, Source>(source_map)) {}

  TVM_DLL SourceMap() : SourceMap(Map<SourceName, Source>()) {}

  void Add(const Source& source);

  SourceMapNode* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<SourceMapNode*>(get_mutable());
  }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SourceMap, ObjectRef, SourceMapNode);
};

}  // namespace parser
}  // namespace tvm

#endif  // TVM_PARSER_SOURCE_MAP_H_
