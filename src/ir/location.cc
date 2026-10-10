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
 * \file location.cc
 * \brief The implementation of the source map data structure.
 */
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/location.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>

#include <algorithm>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  SourceNode::RegisterReflection();
  SourceMapObj::RegisterReflection();
}

ffi::ObjectPtr<SourceNameNode> GetSourceNameNode(const ffi::String& name) {
  // always return pointer as the reference can change as map re-allocate.
  // or use another level of indirection by creating a unique_ptr
  static std::unordered_map<ffi::String, ffi::ObjectPtr<SourceNameNode>> source_map;

  auto sn = source_map.find(name);
  if (sn == source_map.end()) {
    ffi::ObjectPtr<SourceNameNode> n = ffi::make_object<SourceNameNode>();
    source_map[name] = n;
    n->name = std::move(name);
    return n;
  } else {
    return sn->second;
  }
}

ffi::ObjectPtr<SourceNameNode> GetSourceNameNodeByStr(const std::string& name) {
  return GetSourceNameNode(name);
}

SourceName SourceName::Get(const ffi::String& name) { return SourceName(GetSourceNameNode(name)); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SourceNameNode::RegisterReflection();
  // overrride SourceNameNode to serialization mechanism
  refl::TypeAttrDef<SourceNameNode>()
      .def(tvm::ffi::reflection::type_attr::kDataToJson,
           [](const SourceNameNode* node) {
             // simply save as the string
             return node->name;
           })
      .def(tvm::ffi::reflection::type_attr::kDataFromJson, SourceName::Get);
  refl::TypeAttrDef<SourceNameNode>().def(
      refl::type_attr::kRepr, [](SourceName sn, ffi::Function) -> ffi::String {
        std::ostringstream os;
        os << "SourceName(" << sn->name << ", " << static_cast<const void*>(sn.get()) << ")";
        return os.str();
      });

  refl::GlobalDef().def("ir.SourceName", SourceName::Get);
}

namespace {

const ffi::ObjectPtr<UnknownLocNode>& UnknownLocation() {
  static const auto unknown = ffi::make_object<UnknownLocNode>();
  return unknown;
}

}  // namespace

UnknownLoc::UnknownLoc() : Location(ffi::UnsafeInit{}) { data_ = UnknownLocation(); }

SourceLoc::SourceLoc(SourceName source_name, int start_line, int start_column, int end_line,
                     int end_column)
    : Location(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<SourceLocNode>();
  n->source_name = std::move(source_name);
  n->start_line = start_line;
  n->start_column = start_column;
  n->end_line = end_line;
  n->end_column = end_column;
  data_ = std::move(n);
}

SourceLoc SourceLoc::Merge(const SourceLoc& other) const {
  TVM_FFI_ICHECK((*this)->source_name == other->source_name);
  return SourceLoc((*this)->source_name, std::min((*this)->start_line, other->start_line),
                   std::min((*this)->start_column, other->start_column),
                   std::max((*this)->end_line, other->end_line),
                   std::max((*this)->end_column, other->end_column));
}

CallSiteLoc::CallSiteLoc(Location callee, Location caller) : Location(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<CallSiteLocNode>();
  n->callee = std::move(callee);
  n->caller = std::move(caller);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LocationNode::RegisterReflection();
  UnknownLocNode::RegisterReflection();
  SourceLocNode::RegisterReflection();
  CallSiteLocNode::RegisterReflection();
  refl::TypeAttrDef<UnknownLocNode>()
      .def(refl::type_attr::kRepr,
           [](UnknownLoc, ffi::Function) -> ffi::String { return "UnknownLoc()"; })
      .def(refl::type_attr::kDataToJson, [](const UnknownLocNode*) { return nullptr; })
      .def(refl::type_attr::kDataFromJson, [](ffi::AnyView) { return UnknownLoc(); });
  refl::TypeAttrDef<SourceLocNode>().def(
      refl::type_attr::kRepr, [](SourceLoc loc, ffi::Function fn_repr) -> ffi::String {
        std::ostringstream os;
        os << "SourceLoc(" << fn_repr(ffi::AnyView(loc->source_name)).cast<ffi::String>() << ", "
           << loc->start_line << ", " << loc->start_column << ", " << loc->end_line << ", "
           << loc->end_column << ")";
        return os.str();
      });
  refl::TypeAttrDef<CallSiteLocNode>().def(
      refl::type_attr::kRepr, [](CallSiteLoc loc, ffi::Function fn_repr) -> ffi::String {
        return "CallSiteLoc(" + fn_repr(loc->callee).cast<ffi::String>() + ", " +
               fn_repr(loc->caller).cast<ffi::String>() + ")";
      });
  refl::GlobalDef()
      .def("ir.UnknownLoc", []() { return UnknownLoc(); })
      .def("ir.SourceLoc",
           [](SourceName source_name, int start_line, int start_column, int end_line,
              int end_column) {
             return SourceLoc(source_name, start_line, start_column, end_line, end_column);
           })
      .def("ir.CallSiteLoc", [](Location callee, Location caller) {
        return CallSiteLoc(std::move(callee), std::move(caller));
      });
}

/*! \brief Construct a source from a string. */
Source::Source(SourceName src_name, std::string source) {
  auto n = ffi::make_object<SourceNode>();
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

tvm::ffi::String Source::GetLine(int line) {
  VLOG(1) << "Source::GetLine: line=" << line;
  TVM_FFI_ICHECK(line - 1 < static_cast<int64_t>((*this)->line_map.size()))
      << "requested line: " << line << "at index: " << (line - 1)
      << "line_map size: " << (*this)->line_map.size() << "source: " << (*this)->source;

  // Adjust for zero indexing, now have (line_start, line_length);
  auto range = (*this)->line_map.at(line - 1);
  int line_start = range.first;
  int line_length = range.second;
  VLOG(1) << "Source::GetLine: line_start=" << line_start << " line_length=" << line_length;
  // TODO(@jroesch): expose substring on tvm::ffi::String.
  auto line_text = std::string((*this)->source).substr(line_start, line_length);
  VLOG(1) << "Source::GetLine: line_text=" << line_text;
  return line_text;
}

SourceMap::SourceMap(ffi::Map<SourceName, Source> source_map) {
  auto n = ffi::make_object<SourceMapObj>();
  n->source_map = std::move(source_map);
  data_ = std::move(n);
}

void SourceMap::Add(const Source& source) { (*this)->source_map.Set(source->source_name, source); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("SourceMapAdd", [](SourceMap map, ffi::String name, ffi::String content) {
    auto src_name = SourceName::Get(name);
    Source source(src_name, content);
    map.Add(source);
    return src_name;
  });
}

}  // namespace tvm
