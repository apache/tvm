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
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>
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
    const std::initializer_list<std::initializer_list<ShapeTuple::index_type>>& constant_shapes =
        {}) {
  Array<ObjectRef> constants;
  for (const auto& shape : constant_shapes) {
    constants.push_back(MakeConstant(shape));
  }
  Map<String, Array<ObjectRef>> metatable;
  metatable.Set("relay.Constant", constants);
  IRModule mod = ParseModule("string", mod_text, {}, metatable);
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
                                         //  index, kind
      %0 = abs(%x);                      //  3, E
      %1 = nn.relu(%0);                  //  4, E
      nn.relu(%1)                        //  5, E
    }
  )";
  return MakeTestFunction(kMod);
}

Function VariantTestFunction() {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(10, 10), float32]) {
                                         // index, kind
      %0 = abs(%x);                      // 4, E
      %1 = add(%0, %x);                  // 5, E
      shape_of(%1)                       // 6, O
    }
  )";
  return MakeTestFunction(kMod);
}

Function GPT2ExtractOps() {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(1600, 768), float32]) {
                                                                               // index, kind
      %60 = nn.dense(%x, meta[relay.Constant][0] /*(3072, 768)*/, units=3072); // 6,  A
      %61 = add(%60, meta[relay.Constant][1] /*(3072)*/);                      // 8,  B
      %62 = reshape(%61, newshape=[50, 32, 3072]);                             // 9,  I
      %63 = power(%62, 3f);                                                    // 15, B
      %64 = multiply(%63, 0.044715f);                                          // 17, B
      %65 = add(%62, %64);                                                     // 18, B
      %66 = multiply(%65, 0.797885f);                                          // 20, B
      %67 = tanh(%66);                                                         // 21, E
      %68 = multiply(%62, 0.5f);                                               // 11, B
      %69 = add(%67, 1f);                                                      // 23, B
      multiply(%68, %69)                                                       // 24, B
    }
  )";
  return MakeTestFunction(kMod, {{3072, 768}, {3072}});
}

Function GPT2ExtractTuples() {
  constexpr const char* kMod = R"(
    #[version = "0.0.5"]
    def @main(%x: Tensor[(50, 32, 2304), float32]) {
                                                                           // index, kind
      %19 = split(%x, indices_or_sections=[768, 1536], axis=2);            // 6,  I
      %23 = %19.1;                                                         // 7
      %24 = reshape(%23, newshape=[50, 32, 12, 64]);                       // 8,  I
      %35 = %19.2;                                                         // 11
      %36 = reshape(%35, newshape=[50, 32, 12, 64]);                       // 12, I
      %37 = transpose(%36, axes=[0, 2, 1, 3]);                             // 13, I
      %855 = transpose(%24, axes=[0, 2, 1, 3]);                            // 9,  I
      %856 = expand_dims(%855, axis=0);                                    // 10, B
      %857 = expand_dims(%37, axis=0);                                     // 14, B
      %858 = (%856, %857);                                                 // 15, B
      concatenate(%858)                                                    // 16, I
    }
  )";
  return MakeTestFunction(kMod);
}

PartitionSpec StandardSpec(const std::string& spec_name = "test_spec",
                           const std::string& target = "llvm") {
  return PartitionSpec(spec_name, Target(target), {});
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
    const DataflowGraph& graph, const PartitionSpec& spec,
    const std::vector<std::vector<PostDfsIndex>>& index_sets) {
  std::vector<CandidatePartition> candidate_partitions;
  for (const auto& indexes : index_sets) {
    auto subgraph = SubGraph(graph, IndexSet(graph.size(), indexes));
    auto candidate = CandidatePartition(/*rule_name=*/"", subgraph, spec);
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
    ASSERT_EQ(expected_set.count(actual_candidate), 1) << actual_candidate->ToString();
  }
}

void AssertEqual(const Expr& actual, const Expr& expected) {
  ASSERT_TRUE(StructuralEqual()(actual, expected)) << PrettyPrint(actual);
}

TEST(PartitionRule, DFPatternSingleOp) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto rule = DFPatternPartitionRule("relu_pattern", pattern);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    auto expected_candidates = ExpectedCandidates(graph, spec, {{4}, {5}});
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, DFPatternOverlap) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    auto pattern =
        IsOp("nn.relu")({IsOp("nn.relu")({IsWildcard()}) || IsOp("abs")({IsWildcard()})});
    auto rule = DFPatternPartitionRule("relu+abs_pattern", pattern);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    auto expected_candidates = ExpectedCandidates(graph, spec, {{3, 4}, {4, 5}});
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, Composite) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

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
  Expr expected_expr = MakeTestFunction(kExpectedMod);

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = DFPatternPartitionRule("relu_pattern", pattern);
    auto composite_rule = CompositePartitionRule("composite", df_rule);

    auto actual_candidates = ActualCandidates(graph, func, spec, composite_rule);
    auto actual_expr = CandidatePartition::ParallelRewrite(graph, actual_candidates);

    auto expected_candidates = ExpectedCandidates(graph, spec, {{4}, {5}});
    AssertEqual(actual_candidates, expected_candidates);
    AssertEqual(actual_expr, expected_expr);
  }
}

TEST(PartitionRule, PrimitiveTVM) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

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
  Expr expected_expr = MakeTestFunction(kExpectedMod);

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = DFPatternPartitionRule("relu_pattern", pattern);
    auto primitive_rule = PrimitivePartitionRule("primitive", df_rule);

    auto actual_candidates = ActualCandidates(graph, func, spec, primitive_rule);
    auto actual_expr = CandidatePartition::ParallelRewrite(graph, actual_candidates);

    auto expected_candidates = ExpectedCandidates(graph, spec, {{4}, {5}});
    AssertEqual(actual_candidates, expected_candidates);
    AssertEqual(actual_expr, expected_expr);
  }
}

TVM_REGISTER_TARGET_KIND("test_ext_codegen", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

TEST(PartitionRule, PrimitiveExternal) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec("test_ext_codegen", "test_ext_codegen");

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
  Expr expected_expr = MakeTestFunction(kExpectedMod);

  {
    auto pattern = IsOp("nn.relu")({IsWildcard()});
    auto df_rule = DFPatternPartitionRule("relu_pattern", pattern);
    auto primitive_rule = PrimitivePartitionRule("primitive", df_rule);

    auto actual_candidates = ActualCandidates(graph, func, spec, primitive_rule);
    auto actual_expr = CandidatePartition::ParallelRewrite(graph, actual_candidates);

    auto expected_candidates = ExpectedCandidates(graph, spec, {{4}, {5}});
    AssertEqual(actual_candidates, expected_candidates);
    AssertEqual(actual_expr, expected_expr);
  }
}

TEST(PartitionRule, Union) {
  auto func = StandardTestFunction();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    auto abs_pattern = IsOp("abs")({IsWildcard()});
    auto abs_rule = DFPatternPartitionRule("abs_pattern", abs_pattern);
    auto relu_pattern = IsOp("nn.relu")({IsWildcard()});
    auto relu_rule = DFPatternPartitionRule("relu_pattern", relu_pattern);
    auto union_rule = UnionPartitionRule("union", {abs_rule, relu_rule});

    auto actual_candidates = ActualCandidates(graph, func, spec, union_rule);

    auto expected_candidates = ExpectedCandidates(graph, spec, {{3}, {4}, {5}});
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, OpCallByKind) {
  auto func = VariantTestFunction();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    auto rule = OpCallByKindPartitionRule("op_call_by_kind");
    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    auto expected_candidates = ExpectedCandidates(graph, spec, {{4}, {5}});
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, Combine_ByKind) {
  auto func = GPT2ExtractOps();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    // Prime the system by picking out all 11 calls to non-opaque ops.
    auto sub_rule = OpCallByKindPartitionRule("op_call_by_kind");
    // Combine all <= kOutEWiseFusable (A) actual_candidates (ie anything) with downstream
    // <= kBroadcast (B) actual_candidates (ie B or E).
    Array<SimpleCombinerRule> simple_rules;
    simple_rules.push_back(ByKindSimpleCombinerRule(/*upstream_kind=*/kOutEWiseFusable,
                                                    /*downstream_kind=*/kBroadcast));
    Array<CombinerRule> combiner_rules;
    combiner_rules.push_back(AllSimpleCombinerRule("all_simple", std::move(simple_rules)));
    // Build the overall partition rule.
    auto rule = CombinePartitionRule("combine_by_kind_A_B", std::move(sub_rule),
                                     std::move(combiner_rules), /*max_depth=*/3);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    // The original calls.
    std::vector<std::vector<PostDfsIndex>> expected;
    expected.push_back({6});
    expected.push_back({8});
    expected.push_back({9});
    expected.push_back({11});
    expected.push_back({15});
    expected.push_back({17});
    expected.push_back({18});
    expected.push_back({20});
    expected.push_back({21});
    expected.push_back({23});
    expected.push_back({24});

    // nn.dense (A) and the following add (B)
    expected.push_back({6, 8});

    // reshape (I) and the following power or multiply or both
    expected.push_back({9, 11});
    expected.push_back({9, 15});
    expected.push_back({9, 11, 15});

    // reshape (I) and the following power and multiply
    expected.push_back({9, 15, 17});

    // reshape (I) and everything after it to the max depth of 3
    expected.push_back({9, 11, 15, 17});

    // pairs of broadcasts
    expected.push_back({11, 24});  // multiply / multiply
    expected.push_back({15, 17});  // power / multiply
    expected.push_back({17, 18});  // multiply / add
    expected.push_back({18, 20});  // add / multiply
    expected.push_back({20, 21});  // multiply / tanh
    expected.push_back({21, 23});  // tanh / add
    expected.push_back({23, 24});  // add / multiply

    // triples of broadcasts
    expected.push_back({15, 17, 18});  // power / multiply / add
    expected.push_back({17, 18, 20});  // multiply / add / multiply
    expected.push_back({18, 20, 21});  // add / multiply / tanh
    expected.push_back({20, 21, 23});  // multiply / tanh / add
    expected.push_back({21, 23, 24});  // tanh / add / multiply

    auto expected_candidates = ExpectedCandidates(graph, spec, expected);
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, Combine_TupleArg) {
  auto func = GPT2ExtractTuples();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    // Prime the system by picking out all 8 calls to non-opaque ops.
    auto sub_rule = OpCallByKindPartitionRule("op_call_by_kind");
    // Merge args of tuples of <= injective (I) fields into the call's group.
    Array<CombinerRule> combiner_rules;
    combiner_rules.push_back(TupleArgCombinerRule("tuple_arg"));
    // Build the overall partition rule.
    auto rule = CombinePartitionRule("combine_tuple_arg", std::move(sub_rule),
                                     std::move(combiner_rules), /*max_depth=*/3);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    // The original calls
    std::vector<std::vector<PostDfsIndex>> expected;
    expected.push_back({6});
    expected.push_back({8});
    expected.push_back({9});
    expected.push_back({10});
    expected.push_back({12});
    expected.push_back({13});
    expected.push_back({14});
    expected.push_back({16});

    // The concatenate((expand_dims(...), expand_dims(...)) is grouped.
    expected.push_back({10, 14, 15, 16});

    auto expected_candidates = ExpectedCandidates(graph, spec, expected);
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, Combine_TupleProj) {
  auto func = GPT2ExtractTuples();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    // Prime the system by picking out all 8 calls to non-opaque ops.
    auto sub_rule = OpCallByKindPartitionRule("op_call_by_kind");
    // Merge projections from injective groups.
    Array<CombinerRule> combiner_rules;
    combiner_rules.push_back(TupleProjCombinerRule("tuple_proj"));
    // Build the overall partition rule.
    auto rule = CombinePartitionRule("combine_tuple_proj", std::move(sub_rule),
                                     std::move(combiner_rules), /*max_depth=*/3);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    // The original calls
    std::vector<std::vector<PostDfsIndex>> expected;
    expected.push_back({6});
    expected.push_back({8});
    expected.push_back({9});
    expected.push_back({10});
    expected.push_back({12});
    expected.push_back({13});
    expected.push_back({14});
    expected.push_back({16});

    // split / proj 1
    expected.push_back({6, 7});
    // split / proj 2
    expected.push_back({6, 11});
    // split and both projections
    expected.push_back({6, 7, 11});

    auto expected_candidates = ExpectedCandidates(graph, spec, expected);
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, Combine_Constant) {
  auto func = GPT2ExtractOps();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    // Prime the system by picking out all 11 calls to non-opaque ops.
    auto sub_rule = OpCallByKindPartitionRule("op_call_by_kind");
    // Merge constant args into injective groups
    Array<CombinerRule> combiner_rules;
    combiner_rules.push_back(ConstantCombinerRule("constant"));
    // Build the overall partition rule.
    auto rule = CombinePartitionRule("combine_constant", std::move(sub_rule),
                                     std::move(combiner_rules), /*max_depth=*/3);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    // The original calls
    std::vector<std::vector<PostDfsIndex>> expected;
    expected.push_back({6});
    expected.push_back({8});
    expected.push_back({9});
    expected.push_back({11});
    expected.push_back({15});
    expected.push_back({17});
    expected.push_back({18});
    expected.push_back({20});
    expected.push_back({21});
    expected.push_back({23});
    expected.push_back({24});

    // Constant arg to nn.dense
    expected.push_back({5, 6});

    // Constant arg to add
    expected.push_back({7, 8});

    auto expected_candidates = ExpectedCandidates(graph, spec, expected);
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, Combine_Mixed) {
  auto func = GPT2ExtractOps();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    // Prime the system by picking out all 11 calls to non-opaque ops.
    auto sub_rule = OpCallByKindPartitionRule("op_call_by_kind");

    // Mimic the FuseOps rules.
    Array<SimpleCombinerRule> simple_rules;
    simple_rules.push_back(ByKindSimpleCombinerRule(kOutEWiseFusable, kBroadcast));
    simple_rules.push_back(ByKindSimpleCombinerRule(kBroadcast, kCommReduce));
    simple_rules.push_back(ByKindSimpleCombinerRule(kInjective, kInjective));
    Array<CombinerRule> combiner_rules;
    combiner_rules.push_back(AllSimpleCombinerRule("all_simple", std::move(simple_rules)));

    // Merge constant args into injective groups
    combiner_rules.push_back(ConstantCombinerRule("constant"));

    // Build the overall partition rule.
    auto rule = CombinePartitionRule("combine_mixed", std::move(sub_rule),
                                     std::move(combiner_rules), /*max_depth=*/3);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    // The original calls
    std::vector<std::vector<PostDfsIndex>> expected;
    expected.push_back({6});
    expected.push_back({8});
    expected.push_back({9});
    expected.push_back({11});
    expected.push_back({15});
    expected.push_back({17});
    expected.push_back({18});
    expected.push_back({20});
    expected.push_back({21});
    expected.push_back({23});
    expected.push_back({24});

    // A -> B merging
    expected.push_back({6, 8});
    expected.push_back({9, 11});
    expected.push_back({9, 15});
    expected.push_back({9, 11, 15});
    expected.push_back({9, 15, 17});
    expected.push_back({9, 11, 15, 17});
    expected.push_back({11, 24});
    expected.push_back({15, 17});
    expected.push_back({17, 18});
    expected.push_back({18, 20});
    expected.push_back({20, 21});
    expected.push_back({21, 23});
    expected.push_back({23, 24});
    expected.push_back({15, 17, 18});
    expected.push_back({17, 18, 20});
    expected.push_back({18, 20, 21});
    expected.push_back({20, 21, 23});
    expected.push_back({21, 23, 24});

    // Constant args
    expected.push_back({5, 6});
    expected.push_back({7, 8});

    // B -> R
    expected.push_back({8, 9});
    expected.push_back({8, 9, 11});
    expected.push_back({8, 9, 15});

    // Constant's and A -> B
    expected.push_back({5, 6, 8});
    expected.push_back({5, 6, 7, 8});

    // Constants and B -> R
    expected.push_back({7, 8, 9});
    expected.push_back({7, 8, 9, 11});
    expected.push_back({7, 8, 9, 15});

    auto expected_candidates = ExpectedCandidates(graph, spec, expected);
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, OnlyValid) {
  auto func = GPT2ExtractOps();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    // Prime the system by picking out all 11 calls to non-opaque ops.
    auto sub_rule = OpCallByKindPartitionRule("op_call_by_kind");
    // Combine all <= kOutEWiseFusable (A) actual_candidates (ie anything) with downstream
    // <= kBroadcast (B) actual_candidates (ie B or E).
    Array<SimpleCombinerRule> simple_rules;
    simple_rules.push_back(ByKindSimpleCombinerRule(/*upstream_kind=*/kOutEWiseFusable,
                                                    /*downstream_kind=*/kBroadcast));
    Array<CombinerRule> combiner_rules;
    combiner_rules.push_back(AllSimpleCombinerRule("all_simple", std::move(simple_rules)));
    auto combine_rule = CombinePartitionRule("combine_by_kind_A_B", std::move(sub_rule),
                                             std::move(combiner_rules), /*max_depth=*/3);
    // Only allow up to depth 2, no taps and 1 exit.
    SubGraphConfig config;
    config.allow_taps = false;
    config.max_depth = 2;
    config.max_exits = 1;

    // Build the overall partition rule.
    auto rule = OnlyValidPartitionRule("only_valid", std::move(combine_rule), config);

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    // The original calls.
    std::vector<std::vector<PostDfsIndex>> expected;
    expected.push_back({6});
    expected.push_back({8});
    expected.push_back({9});
    expected.push_back({11});
    expected.push_back({15});
    expected.push_back({17});
    expected.push_back({18});
    expected.push_back({20});
    expected.push_back({21});
    expected.push_back({23});
    expected.push_back({24});

    // nn.dense (A) and the following add (B)
    expected.push_back({6, 8});

    // pairs of broadcasts
    expected.push_back({11, 24});  // multiply / multiply
    expected.push_back({15, 17});  // power / multiply
    expected.push_back({17, 18});  // multiply / add
    expected.push_back({18, 20});  // add / multiply
    expected.push_back({20, 21});  // multiply / tanh
    expected.push_back({21, 23});  // tanh / add
    expected.push_back({23, 24});  // add / multiply

    // The following candidates are filtered out because they have 2 or 3 exits:
    // {9, 11}, {9, 15}, {9,11,15}, {9,15,17}, {15,17,18}, {17,18,20},
    // {18,20,21}, {20,21,23}, {21,23,24}, {9,11,15,17}

    auto expected_candidates = ExpectedCandidates(graph, spec, expected);
    AssertEqual(actual_candidates, expected_candidates);
  }
}

TEST(PartitionRule, Host) {
  auto func = GPT2ExtractTuples();
  auto graph = DataflowGraph(func);
  auto spec = StandardSpec();

  {
    auto rule = HostPartitionRule("host");

    auto actual_candidates = ActualCandidates(graph, func, spec, rule);

    std::vector<std::vector<PostDfsIndex>> expected;

    // Function arg %x
    expected.push_back({0});
    // Operators
    expected.push_back({1});  // concatenate
    expected.push_back({2});  // expand_dims
    expected.push_back({3});  // transpose
    expected.push_back({4});  // reshape
    expected.push_back({5});  // split
    // Tuple projection
    expected.push_back({7});
    expected.push_back({11});
    // Tuple construction
    expected.push_back({15});
    // The overall @main function
    expected.push_back({17});

    auto expected_candidates = ExpectedCandidates(graph, spec, expected);
    AssertEqual(actual_candidates, expected_candidates);
  }
}

}  // namespace
}  // namespace collage
}  // namespace relay
}  // namespace tvm
