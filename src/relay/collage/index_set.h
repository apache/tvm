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
 * \file src/relay/collage/index_set.h
 * \brief Efficient representation of a set of post-dfs indexes.
 */

#ifndef TVM_RELAY_COLLAGE_INDEX_SET_H_
#define TVM_RELAY_COLLAGE_INDEX_SET_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../ir/dataflow_matcher_impl.h"
#include "../ir/indexed_graph.h"

namespace tvm {
namespace relay {
namespace collage {

using IndexSubst = std::unordered_map<size_t, size_t>;

class IndexSet {
 public:
  IndexSet() = default;
  explicit IndexSet(size_t size) : bitvec_(size, false) {}
  IndexSet(size_t size, const std::vector<size_t>& indexes);

  IndexSet operator&(const IndexSet& that) const;
  IndexSet operator|(const IndexSet& that) const;
  IndexSet operator-(const IndexSet& that) const;
  bool AreDisjoint(const IndexSet& that) const;
  bool IsSubset(const IndexSet& that) const;
  bool Intersects(const IndexSet& that) const;

  bool operator[](size_t index) const {
    ICHECK_LT(index, bitvec_.size());
    return bitvec_[index];
  }

  IndexSet& Add(size_t index) {
    ICHECK_LT(index, bitvec_.size());
    bitvec_[index] = true;
    return *this;
  }

  IndexSet Subst(size_t new_size, const IndexSubst& subst) const;

  size_t end_index() const { return bitvec_.size(); }
  size_t PopCount() const;
  bool IsZero() const;
  size_t FirstInsideIndex() const;
  size_t LastInsideIndex() const;
  size_t NextIndex(size_t index) const;
  size_t FirstOutsideIndex() const;
  bool operator==(const IndexSet& that) const;
  bool operator!=(const IndexSet& that) const;
  bool operator<(const IndexSet& that) const;
  size_t hash() const;
  std::string ToString() const;

  struct IndexSetIterator {
    const IndexSet* set;
    size_t i;

    size_t operator*() const {
      ICHECK_LT(i, set->end_index());
      return i;
    }

    const IndexSetIterator& operator++() {
      ICHECK_LT(i, set->end_index());
      i = set->NextIndex(i);
      return *this;
    }

    bool operator==(const IndexSetIterator& that) const {
      ICHECK(set == that.set);
      return i == that.i;
    }

    bool operator!=(const IndexSetIterator& that) const {
      ICHECK(set == that.set);
      return i != that.i;
    }
  };

  IndexSetIterator begin() const { return IndexSetIterator{this, FirstInsideIndex()}; }
  IndexSetIterator end() const { return IndexSetIterator{this, end_index()}; }

 private:
  explicit IndexSet(std::vector<bool> bitvec) : bitvec_(std::move(bitvec)) {}

  std::vector<bool> bitvec_;
};

struct IndexSetEqual {
  bool operator()(const IndexSet& left, const IndexSet& right) const { return left == right; }
};

struct IndexSetHash {
  size_t operator()(const IndexSet& set) const { return set.hash(); }
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_INDEX_SET_H_
