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
 * \file src/relay/collage/candidate_function_cache.cc
 * \brief A cache of the unique global name and costs for partitioned functions.
 */

#include "./candidate_function_cache.h"

namespace tvm {
namespace relay {
namespace collage {

CandidateFunctionCache::Entry& CandidateFunctionCache::GetEntry(const std::string& label,
                                                                const Function& function) {
  auto itr = cache_.find(function);
  if (itr == cache_.end()) {
    String compiler = function->GetAttr<String>(attr::kCompiler, String("tvm")).value();
    std::string global_symbol_name = name_supply_->Fresh({compiler, label});
    GlobalVar global_symbol(std::move(global_symbol_name), function->checked_type());
    itr = cache_.emplace(function, Entry(std::move(global_symbol))).first;
  }
  return itr->second;
}

GlobalVar CandidateFunctionCache::GetGlobalSymbol(const Function& function) {
  return GetEntry(/*label=*/"", function).global_symbol;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
