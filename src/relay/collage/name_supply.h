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
 * \file src/relay/collage/name_supply.h
 * \brief A source of fresh variable names.
 */

#ifndef TVM_RELAY_COLLAGE_NAME_SUPPLY_H_
#define TVM_RELAY_COLLAGE_NAME_SUPPLY_H_

#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace relay {
namespace collage {

/*! \brief A supply of fresh names. */
class NameSupply {
 public:
  explicit NameSupply(std::string prefix) : prefix_(std::move(prefix)) {}

  NameSupply MakeSubNameSupply();

  void Reserve(const std::string& existing) { next_free_index_.emplace(existing, 1); }

  std::string Fresh(const std::initializer_list<std::string>& hints);

 private:
  /*! \brief Prefix for all names. May be empty. */
  std::string prefix_;
  /*! \brief Next unused index for variables with given basename. */
  std::unordered_map<std::string, int> next_free_index_;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_NAME_SUPPLY_H_
