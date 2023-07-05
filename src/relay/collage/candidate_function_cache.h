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
 * \file src/relay/collage/candidate_function_cache.h
 * \brief A cache of the unique global symbol name and cost for partitioned functions.
 */

#ifndef TVM_RELAY_COLLAGE_CANDIDATE_FUNCTION_CACHE_H_
#define TVM_RELAY_COLLAGE_CANDIDATE_FUNCTION_CACHE_H_

#include <tvm/relay/function.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "../transforms/compiler_function_utils.h"
#include "./cost.h"
#include "./name_supply.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief A cache of the unique global symbol and cost for functions extracted to represent
 * partitions. If two functions are structurally equal (which includes equality of their "Compiler"
 * attributes) then they will share the same global symbol and estimated cost. We rely on the
 * function's attributes to distinguish partitions which are structurally the same graph but
 * intended for different targets.
 */
class CandidateFunctionCache : public transform::GlobalSymbolCache {
 public:
  explicit CandidateFunctionCache(std::shared_ptr<NameSupply> name_supply)
      : name_supply_(std::move(name_supply)) {}

  struct Entry {
    GlobalVar global_symbol;
    Cost cost = Cost::Unknown();  // Filled in when have estimated cost.

    explicit Entry(GlobalVar global_symbol) : global_symbol(std::move(global_symbol)) {}
  };

  /*!
   * \brief Returns the unique entry for \p function. If no such entry already exists, create it
   * and assign it a unique global symbol name.
   */
  Entry& GetEntry(const std::string& label, const Function& function);

  GlobalVar GetGlobalSymbol(const Function& function) final;

 private:
  std::shared_ptr<NameSupply> name_supply_;
  std::unordered_map<Function, Entry, StructuralHash, StructuralEqual> cache_;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_CANDIDATE_FUNCTION_CACHE_H_
