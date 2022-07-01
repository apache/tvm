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

#include "../../../src/relay/collage/partition_spec.h"

namespace tvm {
namespace relay {
namespace {

IRModule TestIRModule() {
  constexpr const char* kModel = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
      %0 = abs(%x);                      //  3
      %1 = nn.relu(%0);                  //  4
      nn.relu(%1)                        //  5
    }
  )";
  return parser::ParseModule("string", kModel);
}

std::vector<collage::CandidatePartition> MakeCandidates(
    const collage::DataflowGraph& graph, const runtime::String rule_name,
    const collage::PartitionSpec& spec, const std::vector<std::vector<size_t>> index_sets) {
  std::vector<collage::CandidatePartition> candidate_partitions;
  for (const auto& indexes : index_sets) {
    auto subgraph = collage::SubGraph(graph, collage::IndexSet(graph.size(), indexes));
    auto candidate = collage::CandidatePartition(rule_name, subgraph, spec);
    candidate_partitions.emplace_back(std::move(candidate));
  }
  return candidate_partitions;
}

TEST(PartitionRule, DFPatternSingleOp) {
  IRModule ir_mod = TestIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = collage::DataflowGraph(main);
  Target target("llvm");
  auto spec = collage::PartitionSpec("test_spec", target, {});

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto rule = collage::DFPatternPartitionRule("relu_pattern", pattern);
    auto expected_candidates = MakeCandidates(graph, "relu_pattern", spec, {{4}, {5}});

    auto candidates = rule->AllCandidates(graph, spec);

    ICHECK_EQ(candidates.size(), 2);
    for (size_t i = 0; i < candidates.size(); i++) {
      ICHECK(collage::CandidatePartitionEquals()(candidates[i], expected_candidates[i]));
    }
  }
}

TEST(PartitionRule, DFPatternOverlap) {
  IRModule ir_mod = TestIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = collage::DataflowGraph(main);
  Target target("llvm");
  auto spec = collage::PartitionSpec("test_spec", target, {});

  {
    auto pattern =
        IsOp("nn.relu")({IsOp("nn.relu")({IsWildcard()}) || IsOp("abs")({IsWildcard()})});
    auto rule = collage::DFPatternPartitionRule("relu+abs_pattern", pattern);
    auto expected_candidates = MakeCandidates(graph, "relu+abs_pattern", spec, {{3, 4}, {4, 5}});

    auto candidates = rule->AllCandidates(graph, spec);

    ICHECK_EQ(candidates.size(), 2);
    for (size_t i = 0; i < candidates.size(); i++) {
      ICHECK(collage::CandidatePartitionEquals()(candidates[i], expected_candidates[i]));
    }
  }
}

TEST(PartitionRule, Composite) {
  IRModule ir_mod = TestIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = collage::DataflowGraph(main);
  Target target("llvm");
  auto spec = collage::PartitionSpec("test_spec", target, {});

  {
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
    auto expected_expr =
        Downcast<Function>(parser::ParseModule("string", kExpectedMod)->Lookup("main"));
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = collage::DFPatternPartitionRule("relu_pattern", pattern);
    auto composite_rule = collage::CompositePartitionRule("composite", df_rule);

    auto candidates = composite_rule->AllCandidates(graph, spec);
    auto rewrite_expr = collage::CandidatePartition::ParallelRewrite(graph, candidates);

    ICHECK_EQ(candidates.size(), 2);
    ICHECK(StructuralEqual()(rewrite_expr, expected_expr));
  }
}

TEST(PartitionRule, PrimitiveTVM) {
  IRModule ir_mod = TestIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = collage::DataflowGraph(main);
  Target target("llvm");
  auto spec = collage::PartitionSpec("test_spec", target, {});

  {
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
    auto expected_expr =
        Downcast<Function>(parser::ParseModule("string", kExpectedMod)->Lookup("main"));
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = collage::DFPatternPartitionRule("relu_pattern", pattern);
    auto primitive_rule = collage::PrimitivePartitionRule("primitive", df_rule);

    auto candidates = primitive_rule->AllCandidates(graph, spec);
    auto rewrite_expr = collage::CandidatePartition::ParallelRewrite(graph, candidates);

    ICHECK_EQ(candidates.size(), 2);
    ICHECK(StructuralEqual()(rewrite_expr, expected_expr));
  }
}

TVM_REGISTER_TARGET_KIND("test_ext_codegen", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

TEST(PartitionRule, PrimitiveExternal) {
  IRModule ir_mod = TestIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = collage::DataflowGraph(main);
  Target target("test_ext_codegen");
  auto spec = collage::PartitionSpec("test_ext_codegen", target, {});

  {
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
    auto expected_expr =
        Downcast<Function>(parser::ParseModule("string", kExpectedMod)->Lookup("main"));
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = collage::DFPatternPartitionRule("relu_pattern", pattern);
    auto primitive_rule = collage::PrimitivePartitionRule("primitive", df_rule);

    auto candidates = primitive_rule->AllCandidates(graph, spec);
    auto rewrite_expr = collage::CandidatePartition::ParallelRewrite(graph, candidates);

    ICHECK_EQ(candidates.size(), 2);
    ICHECK(StructuralEqual()(rewrite_expr, expected_expr));
  }
}

TEST(PartitionRule, Union) {
  IRModule ir_mod = TestIRModule();
  auto main = Downcast<Function>(ir_mod->Lookup("main"));
  auto graph = collage::DataflowGraph(main);
  Target target("llvm");
  auto spec = collage::PartitionSpec("test_spec", target, {});

  {
    auto abs_pattern = IsOp("abs")({IsWildcard()});
    auto abs_rule = collage::DFPatternPartitionRule("abs_pattern", abs_pattern);
    auto relu_pattern = IsOp("nn.relu")({IsWildcard()});
    auto relu_rule = collage::DFPatternPartitionRule("relu_pattern", relu_pattern);
    auto union_rule = collage::UnionPartitionRule("union", {abs_rule, relu_rule});

    auto abs_candidates = MakeCandidates(graph, "abs_pattern", spec, {{3}});
    auto relu_candidates = MakeCandidates(graph, "relu_pattern", spec, {{4}, {5}});

    std::vector<collage::CandidatePartition> expected_candidates;
    expected_candidates.insert(expected_candidates.end(), abs_candidates.begin(),
                               abs_candidates.end());
    expected_candidates.insert(expected_candidates.end(), relu_candidates.begin(),
                               relu_candidates.end());

    auto candidates = union_rule->AllCandidates(graph, spec);

    ICHECK_EQ(candidates.size(), expected_candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) {
      ICHECK(collage::CandidatePartitionEquals()(candidates[i], expected_candidates[i]));
    }
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
  auto main = Downcast<Function>(parser::ParseModule("string", kMod)->Lookup("main"));
  auto graph = collage::DataflowGraph(main);
  Target target("llvm");
  auto spec = collage::PartitionSpec("test_spec", target, {});

  {
    auto rule = collage::OpCallByKindPartitionRule("op_call_by_kind");
    auto expected_candidates = MakeCandidates(graph, "op_call_by_kind", spec, {{4}, {5}});

    auto candidates = rule->AllCandidates(graph, spec);

    ICHECK_EQ(candidates.size(), expected_candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) {
      ICHECK(collage::CandidatePartitionEquals()(candidates[i], expected_candidates[i]));
    }
  }
}

}  // namespace
}  // namespace relay
}  // namespace tvm
