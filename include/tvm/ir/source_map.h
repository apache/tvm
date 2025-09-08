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

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/node.h>
#include <tvm/runtime/object.h>

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
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceNameNode>().def_ro("name", &SourceNameNode::name);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceName", SourceNameNode, Object);
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
  TVM_DLL static SourceName Get(const ffi::String& name);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SourceName, ObjectRef, SourceNameNode);
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

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SpanNode>()
        .def_ro("source_name", &SpanNode::source_name)
        .def_ro("line", &SpanNode::line)
        .def_ro("column", &SpanNode::column)
        .def_ro("end_line", &SpanNode::end_line)
        .def_ro("end_column", &SpanNode::end_column);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Span", SpanNode, Object);
};

class Span : public ObjectRef {
 public:
  TVM_DLL Span(SourceName source_name, int line, int end_line, int column, int end_column);

  /*! \brief Merge two spans into one which captures the combined regions. */
  TVM_DLL Span Merge(const Span& other) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Span, ObjectRef, SpanNode);
};

/*!
 * \brief Store a list of spans for an expr generated from mulitple source exprs
 */
class SequentialSpanNode : public SpanNode {
 public:
  /*! \brief The original source list of spans to construct a sequential span. */
  ffi::Array<Span> spans;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SequentialSpanNode>().def_ro("spans", &SequentialSpanNode::spans);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SequentialSpan", SequentialSpanNode, SpanNode);
};

/*!
 * \brief Reference class of SequentialSpanNode.
 * \sa SequentialSpanNode
 */
class SequentialSpan : public Span {
 public:
  TVM_DLL SequentialSpan(ffi::Array<Span> spans);

  TVM_DLL SequentialSpan(std::initializer_list<Span> init);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SequentialSpan, Span, SequentialSpanNode);
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
  ffi::String source;

  /*! \brief A mapping of line breaks into the raw source. */
  std::vector<std::pair<int, int>> line_map;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceNode>()
        .def_ro("source_name", &SourceNode::source_name)
        .def_ro("source", &SourceNode::source);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Source", SourceNode, Object);
};

class Source : public ObjectRef {
 public:
  TVM_DLL Source(SourceName src_name, std::string source);
  TVM_DLL tvm::ffi::String GetLine(int line);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Source, ObjectRef, SourceNode);
};

/*!
 * \brief A mapping from a unique source name to source fragment.
 */
class SourceMap;
/*!
 * \brief Stores locations in frontend source that generated a node.
 */
class SourceMapObj : public Object {
 public:
  /*! \brief The source mapping. */
  ffi::Map<SourceName, Source> source_map;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceMapObj>().def_ro("source_map", &SourceMapObj::source_map);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceMap", SourceMapObj, Object);
};

class SourceMap : public ObjectRef {
 public:
  explicit SourceMap(ffi::Map<SourceName, Source> source_map);

  explicit SourceMap(std::initializer_list<std::pair<SourceName, Source>> source_map)
      : SourceMap(ffi::Map<SourceName, Source>(source_map)) {}

  SourceMap() : SourceMap(ffi::Map<SourceName, Source>()) {}

  void Add(const Source& source);

  SourceMapObj* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<SourceMapObj*>(get_mutable());
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceMap, ObjectRef, SourceMapObj);
};

}  // namespace tvm

#endif  // TVM_IR_SOURCE_MAP_H_
