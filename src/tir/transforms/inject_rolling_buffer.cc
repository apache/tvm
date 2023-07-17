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
 * \file inject_rolling_buffer.cc
 * \brief Inject rolling buffer statements.

    Rolling buffers are buffers where one of the dimensions has been made into
    a circular buffer. Two optimizations are implemented in order to accomplish
    this: sliding window and storage folding. In particular, the sliding window
    optimization is applied to the entire buffer (to avoid recomputing elements)
    and storage folding is then applied to just the rolling dimension.

    Rolling buffers must be inside a loop with only part of the buffer used per
    iteration. The outermost axis will be rolled over.

    For more information, see the RFC:
    https://discuss.tvm.apache.org/t/rfc-introducing-a-rolling-buffer-scheduling-primitive/9836
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

using arith::IntSet;

struct RollingBufferInfo {
  int rolling_axis;
  int rolling_extent;
  std::vector<int> axis_overlaps;
  std::vector<Optional<Var>> axis_iter_vars;
};

class RollingBufferInjector : public StmtExprMutator {
  std::vector<For> for_loops{};
  std::set<Buffer> rolling_buffers{};
  std::map<Buffer, BufferRealize> buffer_to_buffer_realize{};
  std::map<Buffer, std::vector<AttrStmt>> buffer_to_attrs{};
  std::map<Buffer, RollingBufferInfo> rolling_buffer_to_info{};
  // The actual key type is Var, ObjectRef has been used because
  // of the ambiguous overload for 'operator<'
  std::map<ObjectRef, std::vector<BufferRealize>> hoist_buffer_to_for{};

 public:
  RollingBufferInjector() {}

  Stmt Inject(Stmt stmt) { return ConvertSSA(operator()(std::move(stmt))); }

  Stmt VisitStmt_(const ForNode* op) final {
    // Manage the stack of iter_vars
    for_loops.push_back(GetRef<For>(op));

    auto stmt{StmtExprMutator::VisitStmt_(op)};
    op = stmt.as<ForNode>();

    // Manage the stack of iter_vars
    for_loops.pop_back();

    auto it{hoist_buffer_to_for.find(op->loop_var)};
    if (it != hoist_buffer_to_for.end()) {
      // If the loop corresponds to an iter_var that needs a BufferRealize
      // hoisting to its scope, perform the hoisting
      Stmt body{GetRef<For>(op)};
      for (auto realise : it->second) {
        auto attrs{buffer_to_attrs[realise->buffer]};
        Stmt new_realize{BufferRealize(realise->buffer, realise->bounds, realise->condition, body,
                                       realise->span)};
        // The attributes attached to the BufferRealize need hoisting too
        for (auto attr : attrs) {
          if (attr->attr_key == attr::rolling_buffer_scope) {
            continue;
          }
          new_realize = AttrStmt(attr->node, attr->attr_key, attr->value, new_realize, attr->span);
        }
        body = new_realize;
      }
      return body;
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (auto opt = op->node.as<Buffer>()) {
      auto buffer = opt.value();
      // Keep a dictionary associating attribute statements with the buffers
      // they reference. We'll need this if the buffer gets hoisted and we
      // need to hoist all of its attributes at the same time.
      buffer_to_attrs[buffer].push_back(GetRef<AttrStmt>(op));

      if (op->attr_key == attr::rolling_buffer_scope && Downcast<IntImm>(op->value)->value) {
        // If the attribute is indicating that a buffer should be a rolling
        // buffer, then update the rolling_buffers set to include the buffer
        rolling_buffers.insert(buffer);

        auto it{buffer_to_buffer_realize.find(buffer)};
        ICHECK(it != buffer_to_buffer_realize.end())
            << "Rolling buffer injection failed: no BufferRealize found";
        BufferRealize buffer_realize = it->second;

        // If a BufferRealize has been identified as needing to be made into
        // a rolling buffer, begin the analysis.
        std::vector<Optional<Var>> bound_iter_vars{};
        std::vector<int> bound_overlaps{};
        // We use the bound information of the BufferRealize to calculate
        // how we can legally roll
        auto stride{0};
        auto divisor{1};
        Optional<Var> iter_var{};
        for (auto bound : buffer_realize->bounds) {
          divisor = 1;
          if (auto floor_div = bound->min.as<FloorDivNode>()) {
            // Handle the case of fractional strides
            // They take this form: floordiv(hh.outer, 2)
            // Strip the floordiv and keep track of the divisor
            divisor = Downcast<IntImm>(floor_div->b)->value;
            bound = Range::FromMinExtent(floor_div->a, bound->extent, bound->span);
          }
          if (bound->min.as<IntImmNode>()) {
            // If the bound is an int, we can't roll over it
            iter_var = nullptr;
          } else if (auto var = bound->min.as<VarNode>()) {
            // If the bound is just a Var, that implies the stride is 1
            iter_var = GetRef<Var>(var);
            stride = 1;
          } else {
            // Otherwise, it's the iter var multiplied by the stride
            // If not we're in unknown behaviour, so assert
            auto mul = bound->min.as<MulNode>();
            ICHECK(mul) << "Rolling buffer injection failed: the buffer striding is unsupported";
            auto a = mul->a.as<VarNode>();
            ICHECK(a) << "Rolling buffer injection failed: the buffer striding is unsupported";
            auto b = mul->b.as<IntImmNode>();
            ICHECK(b) << "Rolling buffer injection failed: the buffer striding is unsupported";
            iter_var = GetRef<Var>(a);
            stride = b->value;
          }
          stride = std::ceil(static_cast<float>(stride) / divisor);
          bound_iter_vars.push_back(iter_var);
          if (iter_var) {
            bound_overlaps.push_back(Downcast<IntImm>(bound->extent)->value - stride);
          } else {
            bound_overlaps.push_back(0);
          }
        }
        // Pick the outermost iter_var that's mentioned in the bounds
        // to be the rolling axis
        Optional<Var> roll_iter_var{};
        int roll_axis{1};
        for (auto loop : for_loops) {
          auto loop_var{loop->loop_var};
          iter_var = loop_var;

          auto it{std::find_if(
              bound_iter_vars.begin(), bound_iter_vars.end(),
              [&](Optional<Var> var) { return var && (var.get() == loop_var.get()); })};

          if (it != bound_iter_vars.end()) {
            auto i{std::distance(bound_iter_vars.begin(), it)};
            roll_iter_var = loop_var;
            roll_axis = i;
            break;
          }
        }
        // We must have found an axis to roll over
        ICHECK(roll_iter_var) << "Rolling buffer injection failed: no rolling axis found";
        ICHECK(roll_axis != -1) << "Rolling buffer injection failed: no rolling axis found";

        RollingBufferInfo rolling_buffer_info = {
            roll_axis,
            static_cast<int>(Downcast<IntImm>(buffer_realize->bounds[roll_axis]->extent)->value),
            bound_overlaps,
            bound_iter_vars,
        };
        rolling_buffer_to_info[buffer] = rolling_buffer_info;
        Array<Range> new_bounds{};
        auto shape{buffer->shape};
        for (size_t i{0}; i < shape.size(); ++i) {
          auto extent{shape[i]};
          if (static_cast<int>(i) == rolling_buffer_info.rolling_axis) {
            new_bounds.push_back(Range(0, rolling_buffer_info.rolling_extent));
          } else {
            new_bounds.push_back(Range(0, extent));
          }
        }
        BufferRealize new_realize{BufferRealize(buffer, new_bounds, buffer_realize->condition,
                                                buffer_realize->body, buffer_realize->span)};
        hoist_buffer_to_for[iter_var.value()].push_back(new_realize);
      }
    }

    auto stmt{StmtExprMutator::VisitStmt_(op)};
    op = stmt.as<AttrStmtNode>();

    if (auto opt = op->node.as<Buffer>(); opt && rolling_buffers.count(opt.value())) {
      // Remove the attribute statements attached to rolling buffers
      // because they will have been hoisted to the relevant rolling
      // scope
      return op->body;
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    buffer_to_buffer_realize.insert({op->buffer, GetRef<BufferRealize>(op)});

    auto stmt{StmtExprMutator::VisitStmt_(op)};
    op = stmt.as<BufferRealizeNode>();

    if (rolling_buffers.count(op->buffer)) {
      // Remove the original BufferRealize for rolling buffers
      // because they will have been hoisted to the relevant rolling
      // scope
      return op->body;
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto stmt{StmtExprMutator::VisitStmt_(op)};
    op = stmt.as<BufferStoreNode>();

    auto it{rolling_buffer_to_info.find(op->buffer)};
    if (it != rolling_buffer_to_info.end()) {
      auto rolling_buffer_info{it->second};
      std::vector<PrimExpr> indices{};
      // First modify the access indices to use modulo arithmetic
      // for the rolling axis
      for (size_t i{0}; i < op->indices.size(); ++i) {
        auto index{op->indices[i]};
        if (static_cast<int>(i) == rolling_buffer_info.rolling_axis) {
          indices.push_back(FloorMod(index, rolling_buffer_info.rolling_extent));
        } else {
          indices.push_back(index);
        }
      }
      Stmt buffer_store = BufferStore(op->buffer, op->value, indices, op->span);
      // Then wrap the BufferStores in some Ifs to avoid recomputing elements
      for (size_t i{0}; i < rolling_buffer_info.axis_iter_vars.size(); ++i) {
        auto iter_var{rolling_buffer_info.axis_iter_vars[i]};
        if (iter_var && rolling_buffer_info.axis_overlaps[i] > 0) {
          Var var{iter_var.value()};
          const Map<Var, IntSet> dmap{std::make_pair(var, IntSet::Interval(0, 0))};
          auto term_2{arith::Analyzer{}.int_set(op->indices[i], dmap).min()};
          auto condition = Or(LT(var, 1), GE(term_2, rolling_buffer_info.axis_overlaps[i]));
          buffer_store = IfThenElse(likely(condition), buffer_store);
        }
      }
      return buffer_store;
    } else {
      return stmt;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto expr{StmtExprMutator::VisitExpr_(op)};
    op = expr.as<BufferLoadNode>();

    auto it{rolling_buffer_to_info.find(op->buffer)};
    if (it != rolling_buffer_to_info.end()) {
      auto rolling_buffer_info{it->second};
      std::vector<PrimExpr> indices{};
      // Modify the access indices to use modulo arithmetic
      // for the rolling axis
      for (size_t i{0}; i < op->indices.size(); ++i) {
        auto index{op->indices[i]};
        if (static_cast<int>(i) == rolling_buffer_info.rolling_axis) {
          indices.push_back(FloorMod(index, rolling_buffer_info.rolling_extent));
        } else {
          indices.push_back(index);
        }
      }
      return BufferLoad(op->buffer, indices, op->span);
    } else {
      return expr;
    }
  }
};  // namespace tir

namespace transform {

Pass InjectRollingBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = RollingBufferInjector().Inject(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectRollingBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectRollingBuffer").set_body_typed(InjectRollingBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
