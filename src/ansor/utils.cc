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
 * \file ansor/utils.cc
 * \brief Common utilities
 */

#include "utils.h"
#include <tvm/runtime/registry.h>

namespace tvm {
namespace ansor {

NullStream& NullStream::Global() {
  static NullStream stream;
  return stream;
}

const std::vector<std::vector<PrimExpr> >& SplitFactorizationMemo::GetFactorizationSchemes(
    int extent, int n_lengths, int max_innermost_factor) {
  QueryKey key = std::make_tuple(extent, n_lengths, max_innermost_factor);
  auto it = memory_.find(key);
  if (it != memory_.end()) {
    return it->second;
  }

  tmp_stack_.assign(n_lengths, PrimExpr());
  results_ = &memory_[key];
  n_lengths_ = n_lengths;

  DfsEnumerate(0, extent, max_innermost_factor);

  return *results_;
}

void SplitFactorizationMemo::DfsEnumerate(int now, int remaining_lenght, int max_innermost_factor) {
  if (now == n_lengths_) {
    if (tmp_stack_.back().as<IntImmNode>()->value <= max_innermost_factor) {
      results_->push_back(tmp_stack_);
    }
  } else {
    for (const auto& f : GetFactors(remaining_lenght)) {
      tmp_stack_[now] = PrimExpr(f);
      DfsEnumerate(now + 1, remaining_lenght / f, max_innermost_factor);
    }
  }
}

const std::vector<int>& SplitFactorizationMemo::GetFactors(int n) {
  auto it = factor_memory_.find(n);
  if (it != factor_memory_.end()) {
    return it->second;
  }

  std::vector<int>& res = factor_memory_[n];
  int step = n % 2 == 0 ? 1 : 2;
  for (size_t i = 1; i < static_cast<size_t>(std::sqrt(n)) + 1; i += step) {
    if (n % i == 0) {
      res.push_back(i);
      if (n / i != i) {
        res.push_back(n/i);
      }
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

ThreadPool& ThreadPool::Global() {
  static ThreadPool* pool = new ThreadPool();
  static int ct = 0;

  ct = (ct + 1) % ThreadPool::REFRESH_EVERY;

  if (ct == 0) {
    pool->Abort();
    delete pool;
    pool = new ThreadPool();
  }

  if (pool->NumWorkers() == 0) {
    pool->Launch(std::thread::hardware_concurrency());
  }

  return *pool;
}

TVM_REGISTER_GLOBAL("ansor.utils.GetFactorizationSchemes")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int extent = args[0];
  int n_lengths = args[1];
  int max_innermost_factor = args[2];
  SplitFactorizationMemo memo;

  Array<Array<PrimExpr> > result;
  for (const auto& lens : memo.GetFactorizationSchemes(extent, n_lengths, max_innermost_factor)) {
    result.push_back(lens);
  }

  *ret = result;
});

}  // namespace ansor
}  // namespace tvm
