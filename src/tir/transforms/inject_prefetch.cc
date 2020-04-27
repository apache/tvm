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
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/arith/bound.h>
#include <tvm/arith/analyzer.h>
#include <unordered_set>

namespace tvm {
namespace tir {

using arith::IntSet;
using arith::DomainTouched;

class PrefetchInjector : public StmtMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    if (op && op->attr_key == attr::prefetch_scope) {
      Buffer buffer = Downcast<Buffer>(op->node);
      CHECK_NE(loop_nest_.size(), 0U);
      Domain domain = DomainTouched(op->body, buffer, true, false);
      Region region;

      auto iter_var = loop_nest_.back().get();
      vectorized_[iter_var] = IntSet::single_point(loop_nest_.back() + op->value);

      for (Range r : domain) {
        if (!r.defined()) {
          LOG(WARNING) << "Cannot decide prefetch region for " << buffer;
          return op->body;
        }
        Range res(EvalSet(r, vectorized_).cover_range(none));
        region.push_back(Range::make_by_min_extent(res->min, res->extent));
      }

      vectorized_.erase(iter_var);

      Stmt prefetch = Prefetch(buffer, region);
      return SeqStmt({prefetch, op->body});
    }
    return ret;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    auto &var = op->loop_var;
    loop_nest_.push_back(var);
    if (op->for_type == ForType::Vectorized) {
      vectorized_[var.get()] = IntSet::interval(op->min, (op->min + op->extent) - 1);
    }
    Stmt ret = StmtMutator::VisitStmt_(op);
    if (op->for_type == ForType::Vectorized) {
      vectorized_.erase(var.get());
    }
    loop_nest_.pop_back();
    return ret;
  }

 private:
  std::vector<Var> loop_nest_;
  std::unordered_map<const VarNode *, IntSet> vectorized_;
  static const Range none;
};

const Range PrefetchInjector::none;

Stmt InjectPrefetch(Stmt stmt) {
  return PrefetchInjector()(std::move(stmt));
}


namespace transform {

Pass InjectPrefetch() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = PrefetchInjector()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectPrefetch", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectPrefetch")
.set_body_typed(InjectPrefetch);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
