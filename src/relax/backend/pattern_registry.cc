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

#include "./pattern_registry.h"

#include "../../support/utils.h"

namespace tvm {
namespace relax {
namespace backend {

PatternRegistryEntry::PatternRegistryEntry(String name, DFPattern pattern,
                                           Map<String, DFPattern> arg_patterns, PackedFunc check) {
  ObjectPtr<PatternRegistryEntryNode> n = make_object<PatternRegistryEntryNode>();
  n->name = std::move(name);
  n->pattern = std::move(pattern);
  n->arg_patterns = std::move(arg_patterns);
  n->check = check;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternRegistryEntryNode);

static std::vector<PatternRegistryEntry>* GetRegistryTable() {
  static std::vector<PatternRegistryEntry> table;
  return &table;
}

void RegisterPatterns(Array<PatternRegistryEntry> entries) {
  auto* table = GetRegistryTable();
  for (const auto& entry : entries) {
    table->push_back(entry);
  }
}

void RemovePatterns(Array<String> names) {
  std::unordered_set<String> name_set{names.begin(), names.end()};

  auto* table = GetRegistryTable();
  table->erase(std::remove_if(table->begin(), table->end(),
                              [&](const PatternRegistryEntry& entry) {
                                return name_set.count(entry->name) > 0;
                              }),
               table->end());
}

Array<PatternRegistryEntry> GetPatternsWithPrefix(const String& prefix) {
  auto* table = GetRegistryTable();
  Array<PatternRegistryEntry> result;
  for (auto it = table->rbegin(); it != table->rend(); ++it) {
    if (support::StartsWith((*it)->name, prefix.data())) {
      result.push_back(*it);
    }
  }
  return result;
}

Optional<PatternRegistryEntry> GetPattern(const String& pattern_name) {
  auto* table = GetRegistryTable();
  for (auto it = table->rbegin(); it != table->rend(); ++it) {
    if ((*it)->name == pattern_name) {
      return *it;
    }
  }
  return NullOpt;
}

TVM_REGISTER_GLOBAL("relax.backend.PatternRegistryEntry")
    .set_body_typed([](String name, DFPattern pattern, Map<String, DFPattern> arg_patterns,
                       PackedFunc check) {
      return PatternRegistryEntry(name, pattern, arg_patterns, check);
    });
TVM_REGISTER_GLOBAL("relax.backend.RegisterPatterns").set_body_typed(RegisterPatterns);
TVM_REGISTER_GLOBAL("relax.backend.RemovePatterns").set_body_typed(RemovePatterns);
TVM_REGISTER_GLOBAL("relax.backend.GetPatternsWithPrefix").set_body_typed(GetPatternsWithPrefix);
TVM_REGISTER_GLOBAL("relax.backend.GetPattern").set_body_typed(GetPattern);

}  // namespace backend
}  // namespace relax
}  // namespace tvm
