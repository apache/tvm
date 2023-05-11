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

#include "../../../src/relay/collage/candidate_partition.h"

#include <gtest/gtest.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>
#include <tvm/relay/transform.h>

#include "../../../../src/relay/collage/mock_cost_estimator.h"
#include "../../../src/relay/collage/partition_spec.h"

namespace tvm {
namespace relay {
namespace collage {
namespace {

// NOTE: CandidatePartition::ParallelRewrite is effectively tested in partition_rule_test.cc
// so not re-tested here. The only other non-trivial code is CandidatePartition::EstimateCost

Function MakeTestFunction(const std::string& mod_text) {
  IRModule mod = ParseModule("string", mod_text, {}, {});
  mod = transform::CapturePostDfsIndexInSpans()(mod);
  auto func = Downcast<Function>(mod->Lookup("main"));
  LOG(INFO) << "------- input function -------";
  LOG(INFO) << PrettyPrint(func);
  LOG(INFO) << "------------------------------";
  return func;
}

PartitionSpec StandardSpec() { return PartitionSpec("test_spec", Target("llvm"), {}); }

String AlwaysInvalid(const Function& function) { return "invalid"; }

PartitionSpec AlwaysInvalidSpec() {
  return PartitionSpec("test_spec", Target("llvm"), {}, AlwaysInvalid);
}

/*!
 * \brief Returns candidate containing nodes with given \p indexes wrapped within a
 * "Primitive" and "Compiler" function.
 */
CandidatePartition MakeCandidate(const DataflowGraph& graph, const PartitionSpec& spec,
                                 const std::vector<PostDfsIndex>& indexes) {
  IndexSet inside(graph.size(), indexes);
  SubGraph inner_sub_graph(graph, inside);
  FunctionAttrsMap attrs_map;
  attrs_map.Set(attr::kPrimitive, Integer(1));
  attrs_map.Set(attr::kCompiler, String("llvm"));
  NestedSubGraph nested_sub_graph(inner_sub_graph, attrs_map);
  SubGraph outer_sub_graph(graph, inside, inner_sub_graph->kind_, inner_sub_graph->label_,
                           {nested_sub_graph});
  return CandidatePartition(/*rule_name=*/"", outer_sub_graph, spec);
}

CostEstimator StandardEstimator() {
  Map<String, Integer> target_costs;
  target_costs.Set("llvm", 3);
  return MockCostEstimator(std::move(target_costs));
}

CostEstimator AlternateEstimator() {
  Map<String, Integer> target_costs;
  target_costs.Set("llvm", 7);
  return MockCostEstimator(std::move(target_costs));
}

std::shared_ptr<CandidateFunctionCache> Cache() {
  return std::make_shared<CandidateFunctionCache>(std::make_shared<NameSupply>("test"));
}

TEST(CandidatePartition, EstimateCost_Simple) {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  3
      %1 = nn.relu(%0);                  //  4
      nn.relu(%1)                        //  5
    }
  )";
  auto func = MakeTestFunction(kMod);
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();
  auto candidate = MakeCandidate(graph, spec, {3, 4});
  auto estimator = StandardEstimator();
  auto cache = Cache();

  {
    auto cost = candidate->EstimatedCost(graph, estimator, cache);
    ASSERT_TRUE(cost.is_value());
    // cost is 3 for nn.rulu plus 3 * 0.9 for the nested abs
    ASSERT_EQ(cost.value(), 5.7);
  }
}

TEST(CandidatePartition, EstimateCost_AlreadyCached) {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  3
      %1 = nn.relu(%0);                  //  4
      nn.relu(%1)                        //  5
    }
  )";
  auto func = MakeTestFunction(kMod);
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();
  auto candidate = MakeCandidate(graph, spec, {3, 4});
  candidate->cost_ = Cost::Value(42.0);
  auto estimator = StandardEstimator();
  auto cache = Cache();

  {
    auto cost = candidate->EstimatedCost(graph, estimator, cache);
    ASSERT_TRUE(cost.is_value());
    ASSERT_EQ(cost.value(), 42.0);
  }
}

TEST(CandidatePartition, EstimateCost_Invalid) {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  3
      %1 = nn.relu(%0);                  //  4
      nn.relu(%1)                        //  5
    }
  )";
  auto func = MakeTestFunction(kMod);
  auto graph = DataflowGraph(func);
  auto spec = AlwaysInvalidSpec();
  auto candidate = MakeCandidate(graph, spec, {3, 4});
  auto estimator = StandardEstimator();
  auto cache = Cache();

  {
    auto cost = candidate->EstimatedCost(graph, estimator, cache);
    ASSERT_TRUE(cost.is_invalid());
  }
}

TEST(CandidatePartition, EstimateCost_Cached) {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  4
      %1 = nn.relu(%0);                  //  5
      %2 = abs(%1);                      //  6
      %3 = nn.relu(%2);                  //  7
      add(%1, %3)                        //  8
    }
  )";
  auto func = MakeTestFunction(kMod);
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();
  auto candidateA = MakeCandidate(graph, spec, {4, 5});
  auto candidateB = MakeCandidate(graph, spec, {6, 7});
  auto standard_estimator = StandardEstimator();
  auto alternate_estimator = AlternateEstimator();
  auto cache = Cache();

  {
    // First candidate estimated as per usual.
    auto costA = candidateA->EstimatedCost(graph, standard_estimator, cache);
    ASSERT_TRUE(costA.is_value());
    ASSERT_EQ(costA.value(), 5.7);

    // Second candidate is structurally equal to first, so reuse first's cost even though
    // estimator has different weights.
    auto costB = candidateB->EstimatedCost(graph, alternate_estimator, cache);
    ASSERT_TRUE(costB.is_value());
    ASSERT_EQ(costB.value(), costA.value());
  }
}

TEST(CandidatePartition, EstimateCost_EtaExpandTuples) {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  3
      %1 = nn.relu(%0);                  //  5
      %2 = (%0, %1);                     //  6
      concatenate(%2)                    //  7
    }
  )";
  auto func = MakeTestFunction(kMod);
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();
  auto candidate = MakeCandidate(graph, spec, {7});
  auto estimator = StandardEstimator();
  auto cache = Cache();

  {
    auto cost = candidate->EstimatedCost(graph, estimator, cache);
    ASSERT_TRUE(cost.is_value());
    ASSERT_EQ(cost.value(), 3);
  }
}

}  // namespace
}  // namespace collage
}  // namespace relay
}  // namespace tvm
