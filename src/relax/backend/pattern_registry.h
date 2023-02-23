
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
 * \file relax/backend/contrib/pattern_registry.h
 * \brief Functions related to registering and retrieving patterns for
 * functions handled by backends.
 */
#ifndef TVM_RELAX_BACKEND_PATTERN_REGISTRY_H_
#define TVM_RELAX_BACKEND_PATTERN_REGISTRY_H_

#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relax {
namespace backend {

class PatternRegistryEntryNode : public Object {
 public:
  String name;
  DFPattern pattern;
  Map<String, DFPattern> arg_patterns;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("pattern", &pattern);
    v->Visit("arg_patterns", &arg_patterns);
  }

  static constexpr const char* _type_key = "relax.backend.PatternRegistryEntry";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternRegistryEntryNode, Object);
};

class PatternRegistryEntry : public ObjectRef {
 public:
  PatternRegistryEntry(String name, DFPattern pattern, Map<String, DFPattern> arg_patterns);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PatternRegistryEntry, ObjectRef,
                                            PatternRegistryEntryNode);
};

void RegisterPatterns(Array<PatternRegistryEntry> entries);

Array<PatternRegistryEntry> GetPatternsWithPrefix(const String& prefix);

Optional<PatternRegistryEntry> GetPattern(const String& pattern_name);

}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_PATTERN_REGISTRY_H_
