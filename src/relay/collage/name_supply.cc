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
 * \file src/relay/collage/name_supply.cc
 * \brief A source of fresh variable names.
 */

#include "./name_supply.h"

#include <algorithm>
#include <sstream>

namespace tvm {
namespace relay {
namespace collage {

namespace {
void AppendCSafe(bool* first, std::ostringstream& os, const std::string& str) {
  for (size_t i = 0; i < str.size(); ++i) {
    const char c = str[i];
    if (i == 0 && first && (!std::isalpha(c) && c != '_')) {
      os << "_";
    }
    if (c == '_' || std::isalnum(c)) {
      os << c;
    } else {
      os << "_";
    }
    *first = false;
  }
}
}  // namespace

NameSupply NameSupply::MakeSubNameSupply() {
  NameSupply result(prefix_);
  for (const auto& kv : next_free_index_) {
    result.next_free_index_.emplace(kv.first, kv.second);
  }
  return result;
}

std::string NameSupply::Fresh(const std::initializer_list<std::string>& hints) {
  std::ostringstream os;
  bool first = true;
  bool need_sep = false;
  if (!prefix_.empty()) {
    AppendCSafe(&first, os, prefix_);
    need_sep = true;
  }
  for (const auto& hint : hints) {
    if (hint.empty()) {
      continue;
    }
    if (need_sep) {
      os << "_";
    }
    AppendCSafe(&first, os, hint);
    need_sep = true;
  }
  std::string name = os.str();
  auto itr = next_free_index_.find(name);
  if (itr == next_free_index_.end()) {
    next_free_index_.emplace(name, 1);
  } else {
    os << "_" << itr->second++;
    name = os.str();
  }
  return name;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
