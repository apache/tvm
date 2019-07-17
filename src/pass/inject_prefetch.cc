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
 * \file inject_prefetch.cc
 */
// Inject prefetch op in HalideIR
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <unordered_set>

namespace tvm {
namespace ir {

using arith::IntSet;
using arith::DomainTouched;

class PrefetchInjector : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    Stmt ret = IRMutator::Mutate_(op, s);
    op = ret.as<AttrStmt>();
    if (op && op->attr_key == attr::prefetch_scope) {
      Tensor ts(op->node.node_);
      CHECK_NE(loop_nest_.size(), 0U);
      Domain domain = DomainTouched(op->body, ts, true, false);
      Region region;

      auto iter_var = loop_nest_.back().get();
      vectorized_[iter_var] = IntSet::single_point(loop_nest_.back() + op->value);

      for (Range r : domain) {
        if (!r.defined()) {
          LOG(WARNING) << "Cannot decide prefetch region for " << ts;
          return op->body;
        }
        Range res(EvalSet(r, vectorized_).cover_range(none));
        region.push_back(Range::make_by_min_extent(res->min, res->extent));
      }

      vectorized_.erase(iter_var);

      Stmt prefetch = Prefetch::make(ts->op, ts->value_index, ts->dtype, region);
      return Block::make(prefetch, op->body);
    }
    return ret;
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    auto &var = op->loop_var;
    loop_nest_.push_back(var);
    if (op->for_type == ForType::Vectorized) {
      vectorized_[var.get()] = IntSet::interval(op->min, (op->min + op->extent) - 1);
    }
    Stmt ret = IRMutator::Mutate_(op, s);
    if (op->for_type == ForType::Vectorized) {
      vectorized_.erase(var.get());
    }
    loop_nest_.pop_back();
    return ret;
  }

 private:
  std::vector<VarExpr> loop_nest_;
  std::unordered_map<const Variable *, IntSet> vectorized_;
  static const Range none;
};

const Range PrefetchInjector::none;

Stmt InjectPrefetch(Stmt stmt) {
  return PrefetchInjector().Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
