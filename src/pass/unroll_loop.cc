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
 *  Loop unrolling as in Halide pipeline.
 * \file unroll_loop.cc
 */
// Unrolls the loop as in Halide pipeline.
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

class LoopUnroller : public IRMutator {
 public:
  explicit LoopUnroller(int auto_max_step,
                        int auto_max_depth,
                        int auto_max_extent,
                        bool explicit_unroll)
      : auto_max_step_(auto_max_step),
        auto_max_depth_(auto_max_depth),
        auto_max_extent_(auto_max_extent),
        explicit_unroll_(explicit_unroll) {
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& stmt) final {
    if (op->attr_key == "pragma_auto_unroll_max_step") {
      int value = 0;
      CHECK(arith::GetConstInt(op->value, &value));
      std::swap(value, auto_max_step_);
      Stmt ret = this->Mutate(op->body);
      std::swap(value, auto_max_step_);
      return ret;
    } else if (op->attr_key == "pragma_unroll_explicit") {
      int value = 0;
      CHECK(arith::GetConstInt(op->value, &value));
      bool explicit_unroll = value;
      std::swap(explicit_unroll, explicit_unroll_);
      Stmt ret = this->Mutate(op->body);
      std::swap(explicit_unroll, explicit_unroll_);
      return ret;
    } else {
      return IRMutator::Mutate_(op, stmt);
    }
  }

  Stmt Mutate_(const For* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<For>();
    int value = GetExtent(op);
    // condition for auto unroll
    bool auto_unroll = (
        op->for_type == ForType::Serial &&
        value >= 0 &&
        normal_loop_depth_ == 0 &&
        unroll_depth_ <= auto_max_depth_);

    auto_unroll = auto_unroll && (
        value * step_count_ <= auto_max_step_||
        value <= auto_max_extent_);

    if (op->for_type == ForType::Unrolled) {
      CHECK_GE(value, 0)
          << "Cannot unroll non-constant loop";
      auto_unroll = true;
    }

    if (auto_unroll) {
      step_count_  *=  value;
      unroll_depth_ += 1;
    } else {
      normal_loop_depth_ += 1;
    }

    if ((auto_unroll && explicit_unroll_) ||
        // unroll loops with extent = 1, no matter how many steps in body
        (0 <= value && value <= auto_max_extent_ && auto_max_extent_ == 1)) {
      return Unroll(op);
    } else {
      if (auto_unroll) {
        if (op->for_type != ForType::Unrolled) {
          return For::make(
              op->loop_var, op->min, op->extent,
              ForType::Unrolled, op->device_api, op->body);
        }
      }
      return stmt;
    }
  }

  Stmt Mutate_(const Store* op, const Stmt& stmt) final {
    ++step_count_;
    return IRMutator::Mutate_(op, stmt);
  }

  Stmt Mutate_(const Evaluate* op, const Stmt& stmt) final {
    ++step_count_;
    return IRMutator::Mutate_(op, stmt);
  }

  Stmt Mutate_(const Block* op, const Stmt& stmt) final {
    Stmt first = this->Mutate(op->first);
    // cleanup state
    int step_count = step_count_;
    int unroll_depth = unroll_depth_;
    int normal_loop_depth = normal_loop_depth_;
    step_count_ = 0;
    unroll_depth_ = 0;
    normal_loop_depth_ = 0;
    // work on rest part
    Stmt rest = this->Mutate(op->rest);
    step_count_ += step_count;
    normal_loop_depth_ = std::max(normal_loop_depth, normal_loop_depth_);
    unroll_depth_ = std::max(unroll_depth_, unroll_depth);
    if (first.same_as(op->first) &&
        rest.same_as(op->rest)) {
      return stmt;
    } else {
      return Block::make(first, rest);
    }
  }

  Stmt Unroll(const For* op) {
    int value = GetExtent(op);
    // For loop must have a constant integer extent
    CHECK_NE(value, -1) << "loop doesn't have a constant integer extent";
    if (value == 0) return Evaluate::make(0);
    Stmt body = op->body;
    Map<Var, Expr> vmap;
    Stmt unrolled;
    for (int i = 0; i < value; ++i) {
      Var lv(op->loop_var.node_);
      vmap.Set(lv, op->min + make_const(op->loop_var.type(), i));
      Stmt step = Substitute(body, vmap);
      if (unrolled.defined()) {
        unrolled = Block::make(unrolled, step);
      } else {
        unrolled = step;
      }
    }
    return unrolled;
  }

 private:
  // returns the extent of the loop if it's a constant integer, otherwise return -1
  int GetExtent(const For* op) {
    // constant folding.
    Expr extent = ir::Simplify(op->extent);
    const IntImm  *v1 = extent.as<IntImm>();
    const UIntImm *v2 = extent.as<UIntImm>();
    int value = -1;
    if (v1 != nullptr) {
      value = static_cast<int>(v1->value);
    }
    if (v2 != nullptr) {
      value = static_cast<int>(v2->value);
    }
    return value;
  }

  // maximum number of step to perform auto unroll.
  int auto_max_step_;
  int auto_max_depth_;
  // max extent of loop to auto unroll
  // this not not count the total steps, only count the number of loops
  int auto_max_extent_;
  bool explicit_unroll_;
  // Number of normal loops in scope
  int normal_loop_depth_{0};
  // number of unrolled cases in current scope.
  int unroll_depth_{0};
  // Number of total steps unrolled
  int step_count_{0};
};


Stmt UnrollLoop(Stmt stmt,
                int auto_max_step,
                int auto_max_depth,
                int auto_max_extent,
                bool explicit_unroll) {
  Stmt ret = LoopUnroller(
      auto_max_step,
      auto_max_depth,
      auto_max_extent,
      explicit_unroll).Mutate(stmt);
  if (!ret.same_as(stmt)) {
    return ConvertSSA(ret);
  } else {
    return ret;
  }
}

Stmt UnrollLoopExplicitly(Stmt stmt) {
  const For* op = stmt.as<For>();
  if (!op) {
    LOG(FATAL) << "attempted to unroll a non-loop statement";
  }
  return LoopUnroller(0, 0, 0, false).Unroll(op);
}

}  // namespace ir
}  // namespace tvm
