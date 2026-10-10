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
 * \file location.h
 * \brief Source locations and source-text lookup.
 */
#ifndef TVM_IR_LOCATION_H_
#define TVM_IR_LOCATION_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {

/*!
 * \brief The source name in the Location
 * \sa SourceNameNode, Location
 */
class SourceName;
/*!
 * \brief The name of a source fragment.
 */
class SourceNameNode : public ffi::Object {
 public:
  /*! \brief The source name. */
  ffi::String name;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceNameNode>().def_ro("name", &SourceNameNode::name);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceName", SourceNameNode, ffi::Object);
};

/*!
 * \brief The source name of a file loc.
 * \sa SourceNameNode, Location
 */
class SourceName : public ffi::ObjectRef {
 public:
  /*!
   * \brief Get an SourceName for a given operator name.
   *  Will raise an error if the source name has not been registered.
   * \param name Name of the operator.
   * \return SourceName valid throughout program lifetime.
   */
  TVM_DLL static SourceName Get(const ffi::String& name);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SourceName, ffi::ObjectRef, SourceNameNode);
};

/*! \brief Base class for immutable source-location metadata. */
class LocationNode : public ffi::Object {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<LocationNode>(); }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Location", LocationNode, ffi::Object);
};

/*! \brief A source location, defaulting to the canonical UnknownLoc. */
class Location : public ffi::ObjectRef {
 public:
  TVM_DLL Location();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Location, ffi::ObjectRef, LocationNode);
};

/*! \brief No source-location information is available. */
class UnknownLocNode : public LocationNode {
 public:
  static void RegisterReflection() { ffi::reflection::ObjectDef<UnknownLocNode>(); }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.UnknownLoc", UnknownLocNode, LocationNode);
};

/*! \brief Reference to the shared immutable unknown location. */
class UnknownLoc : public Location {
 public:
  TVM_DLL UnknownLoc();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(UnknownLoc, Location, UnknownLocNode);
};

/*! \brief A range in one source, retaining frontend coordinate conventions. */
class SourceLocNode : public LocationNode {
 public:
  /*! \brief The source name. */
  SourceName source_name;
  /*! \brief The starting line number. */
  int start_line;
  /*! \brief The starting column offset. */
  int start_column;
  /*! \brief The ending line number. */
  int end_line;
  /*! \brief The ending column offset. */
  int end_column;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceLocNode>()
        .def_ro("source_name", &SourceLocNode::source_name)
        .def_ro("start_line", &SourceLocNode::start_line)
        .def_ro("start_column", &SourceLocNode::start_column)
        .def_ro("end_line", &SourceLocNode::end_line)
        .def_ro("end_column", &SourceLocNode::end_column);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceLoc", SourceLocNode, LocationNode);
};

/*! \brief A concrete source range. */
class SourceLoc : public Location {
 public:
  TVM_DLL SourceLoc(SourceName source_name, int start_line, int start_column, int end_line,
                    int end_column);

  /*! \brief Merge two ranges in the same source. */
  TVM_DLL SourceLoc Merge(const SourceLoc& other) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceLoc, Location, SourceLocNode);
};

/*! \brief The location of a callee together with its caller's provenance. */
class CallSiteLocNode : public LocationNode {
 public:
  Location callee;
  Location caller;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallSiteLocNode>()
        .def_ro("callee", &CallSiteLocNode::callee)
        .def_ro("caller", &CallSiteLocNode::caller);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.CallSiteLoc", CallSiteLocNode, LocationNode);
};

/*! \brief Explicit inline callee/caller provenance. */
class CallSiteLoc : public Location {
 public:
  TVM_DLL CallSiteLoc(Location callee, Location caller);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CallSiteLoc, Location, CallSiteLocNode);
};

/*! \brief A program source in any language.
 *
 * Could represent the source from an ML framework or a source
 * representing a tvm::IRModule.
 */
class Source;

class SourceNode : public ffi::Object {
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Source", SourceNode, ffi::Object);
};

class Source : public ffi::ObjectRef {
 public:
  TVM_DLL Source(SourceName src_name, std::string source);
  TVM_DLL tvm::ffi::String GetLine(int line);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Source, ffi::ObjectRef, SourceNode);
};

/*!
 * \brief A mapping from a unique source name to source fragment.
 */
class SourceMap;
/*!
 * \brief Stores locations in frontend source that generated a node.
 */
class SourceMapObj : public ffi::Object {
 public:
  /*! \brief The source mapping. */
  ffi::Map<SourceName, Source> source_map;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SourceMapObj>().def_ro("source_map", &SourceMapObj::source_map);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SourceMap", SourceMapObj, ffi::Object);
};

class SourceMap : public ffi::ObjectRef {
 public:
  explicit SourceMap(ffi::Map<SourceName, Source> source_map);

  explicit SourceMap(std::initializer_list<std::pair<SourceName, Source>> source_map)
      : SourceMap(ffi::Map<SourceName, Source>(source_map)) {}

  SourceMap() : SourceMap(ffi::Map<SourceName, Source>()) {}

  void Add(const Source& source);

  SourceMapObj* operator->() {
    TVM_FFI_ICHECK(get() != nullptr);
    return static_cast<SourceMapObj*>(get_mutable());
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SourceMap, ffi::ObjectRef, SourceMapObj);
};

}  // namespace tvm

#endif  // TVM_IR_LOCATION_H_
