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
 * \file src/relay/collage/index_set.cc
 * \brief Efficient representation of a set of post-dfs indexes.
 */

#include "./index_set.h"

namespace tvm {
namespace relay {
namespace collage {

// TODO(mbs): These should operate one-word-at-a-time

IndexSet::IndexSet(size_t size, const std::vector<size_t>& indexes) : bitvec_(size, false) {
  for (size_t index : indexes) {
    ICHECK_LT(index, bitvec_.size());
    ICHECK(!bitvec_[index]);
    bitvec_[index] = true;
  }
}

IndexSet IndexSet::operator&(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  std::vector<bool> result(bitvec_.size(), false);
  for (size_t index = 0; index < bitvec_.size(); ++index) {
    result[index] = bitvec_[index] && that.bitvec_[index];
  }
  return IndexSet(result);
}

IndexSet IndexSet::operator|(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  std::vector<bool> result(bitvec_.size(), false);
  for (size_t index = 0; index < bitvec_.size(); ++index) {
    result[index] = bitvec_[index] || that.bitvec_[index];
  }
  return IndexSet(result);
}

IndexSet IndexSet::operator-(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  std::vector<bool> result(bitvec_.size());
  for (size_t index = 0; index < bitvec_.size(); ++index) {
    result[index] = bitvec_[index] && !that.bitvec_[index];
  }
  return IndexSet(result);
}

bool IndexSet::AreDisjoint(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (bitvec_[index] && that.bitvec_[index]) {
      return false;
    }
  }
  return true;
}

bool IndexSet::IsSubset(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (bitvec_[index] && !that.bitvec_[index]) {
      return false;
    }
  }
  return true;
}

bool IndexSet::Intersects(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (bitvec_[index] && that.bitvec_[index]) {
      return true;
    }
  }
  return false;
}

IndexSet IndexSet::Subst(size_t new_size, const IndexSubst& subst) const {
  std::vector<bool> result(new_size, false);
  for (PostDfsIndex index = 0; index < bitvec_.size(); ++index) {
    if (!bitvec_[index]) {
      continue;
    }
    auto itr = subst.find(index);
    ICHECK(itr != subst.end());
    PostDfsIndex new_index = itr->second;
    ICHECK(new_index < new_size);
    ICHECK(!result[new_index]);
    result[new_index] = true;
  }
  return IndexSet(result);
}

size_t IndexSet::PopCount() const {
  size_t n = 0;
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (bitvec_[index]) {
      ++n;
    }
  }
  return n;
}

bool IndexSet::IsZero() const {
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (bitvec_[index]) {
      return false;
    }
  }
  return true;
}

size_t IndexSet::FirstInsideIndex() const {
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (bitvec_[index]) {
      return index;
    }
  }
  return bitvec_.size();
}

size_t IndexSet::LastInsideIndex() const {
  for (size_t i = bitvec_.size(); i > 0; i--) {
    const size_t index = i - 1;
    if (bitvec_[index]) {
      return index;
    }
  }
  return bitvec_.size();
}

size_t IndexSet::NextIndex(size_t index) const {
  ICHECK_LT(index, bitvec_.size());
  for (index++; index < bitvec_.size(); index++) {
    if (bitvec_[index]) {
      return index;
    }
  }
  return bitvec_.size();
}

size_t IndexSet::FirstOutsideIndex() const {
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (!bitvec_[index]) {
      return index;
    }
  }
  return bitvec_.size();
}

bool IndexSet::operator==(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  return bitvec_ == that.bitvec_;
}

bool IndexSet::operator!=(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  return bitvec_ != that.bitvec_;
}

bool IndexSet::operator<(const IndexSet& that) const {
  ICHECK_EQ(bitvec_.size(), that.bitvec_.size());
  for (size_t index = 0; index < bitvec_.size(); index++) {
    if (bitvec_[index] && !that.bitvec_[index]) {
      return true;
    }
    if (!bitvec_[index] && that.bitvec_[index]) {
      return false;
    }
  }
  return false;
}

size_t IndexSet::hash() const {
  std::hash<std::vector<bool>> h;
  return h(bitvec_);
}

std::string IndexSet::ToString() const {
  std::ostringstream os;
  os << "{";
  bool first = true;
  for (size_t start = 0; start < bitvec_.size(); /*no-op*/) {
    if (!bitvec_[start]) {
      ++start;
      continue;
    }
    size_t end;
    for (end = start + 1; end < bitvec_.size() && bitvec_[end]; ++end) {
      /*no-op*/
    }
    if (first) {
      first = false;
    } else {
      os << ",";
    }
    os << start;
    if (end > start + 2) {
      os << ".." << (end - 1);
      start = end;
    } else {
      ++start;
    }
  }
  os << "}";
  return os.str();
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
