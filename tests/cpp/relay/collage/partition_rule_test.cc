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

#include "../../../src/relay/collage/partition_rule.h"

#include <gtest/gtest.h>
#include <tvm/parser/parser.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/transform.h>

#include "../../../src/relay/collage/partition_spec.h"

namespace tvm {
namespace relay {
namespace collage {
namespace {

Constant MakeConstant(std::initializer_list<ShapeTuple::index_type> shape) {
  return Constant(runtime::NDArray::Empty(shape, DataType::Float(32), {kDLCPU, 0}));
}

Function MakeTestFunction(
    const std::string& mod_text,
    std::initializer_list<std::initializer_list<ShapeTuple::index_type>> constant_shapes) {
  Array<ObjectRef> constants;
  for (const auto& shape : constant_shapes) {
    constants.push_back(MakeConstant(shape));
  }
  Map<String, Array<ObjectRef>> metatable;
  metatable.Set("relay.Constant", constants);
  IRModule mod = parser::ParseModule("string", mod_text, {}, metatable);
  mod = transform::CapturePostDfsIndexInSpans()(mod);
  auto func = Downcast<Function>(mod->Lookup("main"));
  LOG(INFO) << "------- input function -------";
  LOG(INFO) << PrettyPrint(func);
  LOG(INFO) << "------------------------------";
  return func;
}

Function StandardTestFunction() {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  3
      %1 = nn.relu(%0);                  //  4
      nn.relu(%1)                        //  5
    }
  )";
  return MakeTestFunction(kMod, /*constant_shapes=*/{});
}

std::vector<CandidatePartition> ActualCandidates(const DataflowGraph& graph, const Function& func,
                                                 const PartitionSpec& spec,
                                                 const PartitionRule& rule) {
  auto candidates = rule->AllCandidates(graph, spec);
  LOG(INFO) << "--------- actual candidates -------------";
  for (const auto& candidate : candidates) {
    LOG(INFO) << candidate->ToString();
  }
  LOG(INFO) << "-----------------------------------------";
  return candidates;
}

std::vector<CandidatePartition> ExpectedCandidates(
    const DataflowGraph& graph, const runtime::String rule_name, const PartitionSpec& spec,
    const std::vector<std::vector<PostDfsIndex>> index_sets) {
  std::vector<CandidatePartition> candidate_partitions;
  for (const auto& indexes : index_sets) {
    auto subgraph = SubGraph(graph, IndexSet(graph.size(), indexes));
    auto candidate = CandidatePartition(rule_name, subgraph, spec);
    candidate_partitions.emplace_back(std::move(candidate));
  }
  return candidate_partitions;
}

void AssertEqual(const std::vector<CandidatePartition>& actual,
                 const std::vector<CandidatePartition>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  std::set<CandidatePartition, CandidatePartitionCompare> actual_set(actual.begin(), actual.end());
  std::set<CandidatePartition, CandidatePartitionCompare> expected_set(expected.begin(),
                                                                       expected.end());
  ASSERT_EQ(actual_set.size(), expected_set.size());
  for (const auto& actual_candidate : actual_set) {
    ASSERT_EQ(expected_set.count(actual_candidate), 1);
  }
}

TEST(PartitionRule, DFPatternSingleOp) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  Target target("llvm");
  auto spec = PartitionSpec("test_spec", target, {});

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto rule = DFPatternPartitionRule("relu_pattern", pattern);
    auto expected_candidates = ExpectedCandidates(graph, "relu_pattern", spec, {{4}, {5}});

    auto candidates = ActualCandidates(graph, func, spec, rule);

    ICHECK_EQ(candidates.size(), 2);
    for (size_t i = 0; i < candidates.size(); i++) {
      ICHECK(CandidatePartitionEquals()(candidates[i], expected_candidates[i]));
    }
  }
}

TEST(PartitionRule, DFPatternOverlap) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  Target target("llvm");
  auto spec = PartitionSpec("test_spec", target, {});

  {
    auto pattern =
        IsOp("nn.relu")({IsOp("nn.relu")({IsWildcard()}) || IsOp("abs")({IsWildcard()})});
    auto rule = DFPatternPartitionRule("relu+abs_pattern", pattern);

    auto candidates = ActualCandidates(graph, func, spec, rule);

    auto expected_candidates =
        ExpectedCandidates(graph, "relu+abs_pattern", spec, {{3, 4}, {4, 5}});
    AssertEqual(candidates, expected_candidates);
  }
}

TEST(PartitionRule, Composite) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  Target target("llvm");
  auto spec = PartitionSpec("test_spec", target, {});

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = DFPatternPartitionRule("relu_pattern", pattern);
    auto composite_rule = CompositePartitionRule("composite", df_rule);

    auto candidates = ActualCandidates(graph, func, spec, composite_rule);
    auto rewrite_expr = CandidatePartition::ParallelRewrite(graph, candidates);

    ICHECK_EQ(candidates.size(), 2);

    constexpr const char* kExpectedMod = R"(
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = abs(%x);
        %1 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Composite="composite") {
          nn.relu(%FunctionVar_01)
        };
        %2 = %1(%0);
        %3 = fn (%FunctionVar_0: Tensor[(10, 10), float32], Composite="composite") {
          nn.relu(%FunctionVar_0)
        };
        %3(%2)
      }
    )";
    Expr expected_expr = MakeTestFunction(kExpectedMod, /*constant_shapes=*/{});
    ICHECK(StructuralEqual()(rewrite_expr, expected_expr));
  }
}

TEST(PartitionRule, PrimitiveTVM) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  Target target("llvm");
  auto spec = PartitionSpec("test_spec", target, {});

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = DFPatternPartitionRule("relu_pattern", pattern);
    auto primitive_rule = PrimitivePartitionRule("primitive", df_rule);

    auto candidates = ActualCandidates(graph, func, spec, primitive_rule);
    auto rewrite_expr = CandidatePartition::ParallelRewrite(graph, candidates);

    ICHECK_EQ(candidates.size(), 2);
    constexpr const char* kExpectedMod = R"(
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = abs(%x);
        %1 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Primitive=1) {
          nn.relu(%FunctionVar_01)
        };
        %2 = %1(%0);
        %3 = fn (%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1) {
          nn.relu(%FunctionVar_0)
        };
        %3(%2)
      }
    )";
    Expr expected_expr = MakeTestFunction(kExpectedMod, /*constant_shapes=*/{});
    ICHECK(StructuralEqual()(rewrite_expr, expected_expr));
  }
}

TVM_REGISTER_TARGET_KIND("test_ext_codegen", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

TEST(PartitionRule, PrimitiveExternal) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  Target target("test_ext_codegen");
  auto spec = PartitionSpec("test_ext_codegen", target, {});

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = DFPatternPartitionRule("relu_pattern", pattern);
    auto primitive_rule = PrimitivePartitionRule("primitive", df_rule);

    auto candidates = ActualCandidates(graph, func, spec, primitive_rule);
    auto rewrite_expr = CandidatePartition::ParallelRewrite(graph, candidates);

    ICHECK_EQ(candidates.size(), 2);
    constexpr const char* kExpectedMod = R"(
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = abs(%x);
        %1 = fn (%FunctionVar_01: Tensor[(10, 10), float32], Primitive=1, Compiler="test_ext_codegen") {
          nn.relu(%FunctionVar_01)
        };
        %2 = %1(%0);
        %3 = fn (%FunctionVar_0: Tensor[(10, 10), float32], Primitive=1, Compiler="test_ext_codegen") {
          nn.relu(%FunctionVar_0)
        };
        %3(%2)
      }
    )";
    Expr expected_expr = MakeTestFunction(kExpectedMod, /*constant_shapes=*/{});
    ICHECK(StructuralEqual()(rewrite_expr, expected_expr));
  }
}

TEST(PartitionRule, Union) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  Target target("llvm");
  auto spec = PartitionSpec("test_spec", target, {});

  {
    auto abs_pattern = IsOp("abs")({IsWildcard()});
    auto abs_rule = DFPatternPartitionRule("abs_pattern", abs_pattern);
    auto relu_pattern = IsOp("nn.relu")({IsWildcard()});
    auto relu_rule = DFPatternPartitionRule("relu_pattern", relu_pattern);
    auto union_rule = UnionPartitionRule("union", {abs_rule, relu_rule});

    auto abs_candidates = ExpectedCandidates(graph, "abs_pattern", spec, {{3}});
    auto relu_candidates = ExpectedCandidates(graph, "relu_pattern", spec, {{4}, {5}});

    auto candidates = ActualCandidates(graph, func, spec, union_rule);

    std::vector<CandidatePartition> expected_candidates;
    expected_candidates.insert(expected_candidates.end(), abs_candidates.begin(),
                               abs_candidates.end());
    expected_candidates.insert(expected_candidates.end(), relu_candidates.begin(),
                               relu_candidates.end());
    AssertEqual(candidates, expected_candidates);
  }
}

TEST(PartitionRule, OpCallByKind) {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  4
      %1 = add(%0, %x);                  //  5
      shape_of(%1)                       //  6
    }
  )";
  auto func = MakeTestFunction(kMod, {});
  auto graph = DataflowGraph(func);
  Target target("llvm");
  auto spec = PartitionSpec("test_spec", target, {});

  {
    auto rule = OpCallByKindPartitionRule("op_call_by_kind");
    auto candidates = ActualCandidates(graph, func, spec, rule);

    auto expected_candidates = ExpectedCandidates(graph, "op_call_by_kind", spec, {{4}, {5}});
    AssertEqual(candidates, expected_candidates);
  }
}

}  // namespace
}  // namespace collage
}  // namespace relay
}  // namespace tvm
