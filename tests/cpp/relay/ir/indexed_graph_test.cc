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

#include "../../../src/relay/ir/indexed_graph.h"

#include <gtest/gtest.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>

namespace tvm {
namespace relay {
namespace {

// A module stolen from onnx/test_forward.py::test_loop which combines functions, recursion,
// control flow, tuples as well as the usual operator calls.
// We include the known post-dfs indexes in comments to help write the tests.
IRModule TestRecursiveIRModule() {
  Device device = {kDLCPU, 0};
  Constant const0(runtime::NDArray::Empty(ShapeTuple({1}), DataType::Int(64), device));
  Constant const1(runtime::NDArray::Empty(ShapeTuple({0, 1}), DataType::Float(32), device));
  Map<String, Array<ObjectRef>> metadata;
  metadata.Set("relay.Constant", Array<ObjectRef>({const0, const1}));
  constexpr const char* kModel = R"(
    #[version = "0.0.5"]
    def @main(%trip_count: int64,                                        // 0
              %cond: bool,                                               // 1
              %y: Tensor[(1), float32])                                  // 2
              -> (Tensor[(1), float32], Tensor[(?, ?), float32]) {
      %17 = (
        let %while_loop = fn (%iter_count: int64,                        // 3
                              %max_count: int64,                         // 4
                              %cond_in: bool,                            // 5
                              %y_in: Tensor[(1), float32],               // 6
                              %scan_out: Tensor[(?, ?), float32])        // 7
                              -> (int64, int64, bool, Tensor[(1), float32], Tensor[(?, ?), float32]) {
          %0 = equal(%cond_in, True);                                    // 11
          %1 = less(%iter_count, %max_count);                            // 13
          %2 = logical_and(%0, %1);                                      // 14
          if (%2) {
            %3 = cast(%iter_count, dtype="float32");                     // 20
            %4 = add(%y_in, %3);                                         // 21
            %5 = less(%4, 5f);                                           // 23
            %6 = squeeze(%5);                                            // 24
            %7 = reshape(%iter_count, newshape=[1]);                     // 29
            %8 = (%7, meta[relay.Constant][0]);                          // 31
            %9 = concatenate(%8);                                        // 32
            %10 = copy(%4);                                              // 36
            %11 = dyn.broadcast_to(%scan_out, %9, shape=None);           // 33
            %12 = expand_dims(%10, axis=0);                              // 37
            %13 = (%11, %12);                                            // 38
            %14 = add(%iter_count, 1i64);                                // 17
            %15 = cast(%6, dtype="bool");                                // 25
            %16 = concatenate(%13);                                      // 39
            %while_loop(%14, %max_count, %15, %4, %16)                   // 40
          } else {
            (%iter_count, %max_count, %cond_in, %y_in, %scan_out)        // 41
          }                                                              // 42
        };                                                               // 43
        %while_loop                                                      // 44
      );                                                                 // 45
      %18 = %17(0i64, %trip_count, %cond, %y, meta[relay.Constant][1]);  // 48
      %19 = %18.3;                                                       // 49
      %20 = %18.4;                                                       // 50
      (%19, %20)                                                         // 51
    }                                                                    // 52
  )";
  return ParseModule("string", kModel, /*init_module=*/{}, metadata);
}

TEST(IndexedGraph, RecursiveExprRegression) {
  IRModule ir_mod = TestRecursiveIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = CreateIndexedGraph(main);
  graph->CheckValid();

  {
    // Dataflow node properties for %4
    auto node = graph->index_to_node(21);
    const auto* call_node = node->ref().as<CallNode>();
    ASSERT_NE(call_node, nullptr);
    const auto* op_node = call_node->op.as<OpNode>();
    ASSERT_NE(op_node, nullptr);
    ASSERT_EQ(op_node->name, "add");

    // 3 inputs (the op itself is an input)
    ASSERT_EQ(node->inputs_.size(), 3);
    ASSERT_EQ(node->inputs_[0]->index_, 15);  // the add op
    ASSERT_EQ(node->inputs_[1]->index_, 6);   // %y_in
    ASSERT_EQ(node->inputs_[2]->index_, 20);  // %3

    // 3 outputs
    ASSERT_EQ(node->outputs_.size(), 3);
    ASSERT_EQ(node->outputs_[0]->index_, 23);  // %5
    ASSERT_EQ(node->outputs_[1]->index_, 36);  // %10
    ASSERT_EQ(node->outputs_[2]->index_, 40);  // recursive %while_loop call

    // In the 'if' basic block
    ASSERT_EQ(node->basic_block_->index_, 42);

    // Dominator 'parent' is recursive call
    ASSERT_EQ(node->dominator_parent_->index_, 40);

    // One dominator child from %3
    ASSERT_EQ(node->dominator_children_.size(), 1);
    ASSERT_EQ(node->dominator_children_[0]->index_, 20);
  }

  {
    // The recursive call to %while_loop does not depend on %while_loop
    auto node = graph->index_to_node(40);
    const auto* call_node = node->ref().as<CallNode>();
    ASSERT_NE(call_node, nullptr);
    const auto* var_node = call_node->op.as<VarNode>();
    ASSERT_NE(var_node, nullptr);
    ASSERT_EQ(var_node->name_hint(), "while_loop");

    ASSERT_EQ(node->inputs_.size(), 5);
    ASSERT_EQ(node->inputs_[0]->index_, 17);  // %14
    ASSERT_EQ(node->inputs_[1]->index_, 4);   // %max_count
    ASSERT_EQ(node->inputs_[2]->index_, 25);  // %15
    ASSERT_EQ(node->inputs_[3]->index_, 21);  // %4
    ASSERT_EQ(node->inputs_[4]->index_, 39);  // %16
  }

  {
    // Downstream nodes of %18
    auto node = graph->index_to_node(48);
    std::unordered_set<const IndexedGraph<Expr>::Node*> downstreams;
    node->AccumulateDownstreamNodes(&downstreams);
    ASSERT_EQ(downstreams.size(), 4);
    for (const auto* downstream : downstreams) {
      ASSERT_TRUE(downstream->index_ >= 49 && downstream->index_ <= 52);
    }
  }

  {
    // Dominates relation for %4
    auto upstream = graph->index_to_node(21);
    // Path 1: 21->23->24->25->40
    // Path 2: 21->36->37->38->39->40
    // Then 40->43
    auto downstream = graph->index_to_node(43);
    ASSERT_TRUE(downstream->Dominates(upstream));
  }
}

// A module with unused let-bound function. The 'add' operator should have no dominator
// since it is used both in the unused function and in the main body.
IRModule TestUnusedLetBoundIRModule() {
  constexpr const char* kModel = R"(
    #[version = "0.0.5"]
    def @main(%x: int64) -> int64 {   // 0
      let %f = fn (                   // 5
        %y: int64                     // 1
      ) {
        add(%x, %y)                   // 3
      };
      if (less(%x, 5i64)) {
        add(%x, 3i64)                 // 10
      } else {
        %x
      }
    }
  )";
  return ParseModule("string", kModel);
}

TEST(IndexedGraph, UnusedLetVars) {
  IRModule ir_mod = TestUnusedLetBoundIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = CreateIndexedGraph(main);
  graph->CheckValid();

  {
    auto node = graph->index_to_node(2);
    const auto* op_node = node->ref().as<OpNode>();
    ICHECK(op_node);
    ICHECK_EQ(op_node->name, "add");
    ICHECK_EQ(node->outputs_.size(), 2);
    ICHECK_EQ(node->outputs_[0]->index_, 3);
    ICHECK_EQ(node->outputs_[1]->index_, 10);
    ICHECK(node->dominator_parent_ == nullptr);
  }
}

}  // namespace
}  // namespace relay
}  // namespace tvm
