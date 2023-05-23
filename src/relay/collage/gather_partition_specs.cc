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
 * \file src/relay/collage/gather_partition_specs.cc
 * \brief Gather the relevant \p PartitionSpecs from the available \p Targets.
 */

#include "./gather_partition_specs.h"

#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

namespace {

PartitionRule MakeCombinePartitionRule(PartitionRule sub_rule, Array<CombinerRule> combiner_rules,
                                       size_t max_depth) {
  if (combiner_rules.empty()) {
    return sub_rule;
  } else {
    return CombinePartitionRule("", std::move(sub_rule), std::move(combiner_rules), max_depth);
  }
}

/*! \brief Returns the primitive combiner rules which mimic TVM's \p FuseOps. */
Array<CombinerRule> TVMCombinerRules() {
  Array<SimpleCombinerRule> simple_rules;
  // Mimic the FuseOps rules.
  simple_rules.push_back(ByKindSimpleCombinerRule(kOutEWiseFusable, kBroadcast));
  simple_rules.push_back(ByKindSimpleCombinerRule(kBroadcast, kCommReduce));
  simple_rules.push_back(ByKindSimpleCombinerRule(kInjective, kInjective));

  Array<CombinerRule> combiner_rules;
  // Fire the simple fusion rules
  combiner_rules.push_back(AllSimpleCombinerRule("combiner", std::move(simple_rules)));
  // Fuse tuple arguments
  combiner_rules.push_back(TupleArgCombinerRule("tuple"));
  // Fuse tuple projection
  combiner_rules.push_back(TupleProjCombinerRule("proj"));

  return combiner_rules;
}

size_t GetMaxDepth(std::string key) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  std::string config_key = "relay.collage." + key;
  Optional<Integer> opt_max_depth = ctxt->GetConfig(config_key, Optional<Integer>());
  ICHECK(opt_max_depth.defined()) << "missing binding for '" << config_key << " in pass context";
  ICHECK(opt_max_depth.value()->value > 0)
      << "invalid value for '" << config_key << " in pass context";
  return static_cast<size_t>(opt_max_depth.value()->value);
}

/*! \brief Returns partition rule mimicking TVM FuseOps. */
PartitionRule MakeTVMPartitionRule() {
  size_t max_depth = GetMaxDepth("tvm_max_depth");
  // Build singleton candidates for all calls to ops <= kOutEWiseFusable.
  OpCallByKindPartitionRule op_call_by_kind("");
  // Combine candidates according to the TVM fusion rules.
  PartitionRule combine =
      MakeCombinePartitionRule(std::move(op_call_by_kind), TVMCombinerRules(), max_depth);
  // Discard invalid candidates.
  SubGraphConfig sub_graph_config;
  sub_graph_config.allow_taps = false;
  sub_graph_config.max_depth = max_depth;
  sub_graph_config.max_exits = 1;
  return OnlyValidPartitionRule("", std::move(combine), sub_graph_config);
  // NOTE: We don't wrap by a "Primitive" since we want to defer making TVM fusion decisions until
  // after running more Relay passes.
}

/*!
 * \brief Returns the fusion style for default compiler.
 */
BYOCStyle DefaultBYOCFusionStyleForCompiler(const String& compiler) {
  if (compiler == "cutlass" || compiler == "cublas" || compiler == "cudnn") {
    return kNoFusionBYOCStyle;
  } else if (compiler == "tensorrt") {
    return kTVMFusionBYOCStyle;
  } else {
    return kArbitraryFusionBYOCStyle;
  }
}

/*!
 * \brief Returns the fusion style for given compiler.
 */
BYOCStyle BYOCFusionStyleForCompiler(const String& compiler) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  std::string config_key = "relay.collage.byoc_fusion_style";
  Optional<Array<String>> byoc_configs = ctxt->GetConfig(config_key, Optional<Array<String>>());
  BYOCStyle byoc_fusion_style = DefaultBYOCFusionStyleForCompiler(compiler);
  if (!byoc_configs) {
    return byoc_fusion_style;
  }
  for (auto config_ : byoc_configs.value()) {
    std::vector<std::string> byoc_cfg = SplitString(config_, ".");
    if (byoc_cfg[0] == compiler) {
      if (byoc_cfg[1] == "NoFusion") {
        byoc_fusion_style = kNoFusionBYOCStyle;
      } else if (byoc_cfg[1] == "TVMFusion") {
        byoc_fusion_style = kTVMFusionBYOCStyle;
      } else if (byoc_cfg[1] == "ArbitraryFusion") {
        byoc_fusion_style = kArbitraryFusionBYOCStyle;
      } else {
        ICHECK(false) << "Invalid fusion name for compiler " << byoc_cfg[0] << " in pass context";
      }
      break;
    }
  }
  return byoc_fusion_style;
}

/*!
 * \brief Returns the primitive combiner rules which allow for any touching candidates
 * to be fused provided they don't have kind \p kOpaque.
 */
Array<CombinerRule> BYOCCombinerRules(const String& compiler) {
  Array<SimpleCombinerRule> simple_rules;
  Array<CombinerRule> combiner_rules;
  switch (BYOCFusionStyleForCompiler(compiler)) {
    case kNoFusionBYOCStyle:
      break;
    case kTVMFusionBYOCStyle:
      // Conservatively assume the BYOC toolchain follows the same rules as for TVM's FuseOps.
      simple_rules.push_back(ByKindSimpleCombinerRule(kOutEWiseFusable, kBroadcast));
      simple_rules.push_back(ByKindSimpleCombinerRule(kBroadcast, kCommReduce));
      simple_rules.push_back(ByKindSimpleCombinerRule(kInjective, kInjective));
      combiner_rules.push_back(AllSimpleCombinerRule("combiner", std::move(simple_rules)));
      break;
    case kArbitraryFusionBYOCStyle:
      // Just try all combinations up to the max_depth limit.
      simple_rules.push_back(ByKindSimpleCombinerRule(kOutEWiseFusable, kOutEWiseFusable));
      combiner_rules.push_back(AllSimpleCombinerRule("combiner", std::move(simple_rules)));
      break;
  }
  return combiner_rules;
}

/*!
 * \brief Returns partition rule mimicking one entry in the patterns list passed to the
 * MergeComposite pass.
 */
PartitionRule MakeLabelledDFPatternPartitionRule(
    const std::string& compiler, String rule_name, DFPattern dataflow_pattern,
    TPatternPredicate predicate = DefaultPatternPredicate) {
  DFPatternPartitionRule patterns("", std::move(dataflow_pattern), std::move(predicate));
  return CompositePartitionRule(std::move(rule_name), std::move(patterns));
}

/*!
 * \brief Returns partition rule mimicking
 * MergeComposite/AnnotateTarget/MergeCompilerRegions/PartitionGraph passes for "compiler"
 * attribute of \p target.
 */
PartitionRule MakePatternBYOCPartitionRule(const std::string& compiler,
                                           Array<PartitionRule> sub_rules) {
  size_t max_depth = GetMaxDepth("byoc_max_depth");
  // Union all the individual pattern rules.
  UnionPartitionRule unioned("", std::move(sub_rules));
  PartitionRule combine =
      MakeCombinePartitionRule(std::move(unioned), BYOCCombinerRules(compiler), max_depth);
  // Ignore invalid candidates.
  SubGraphConfig sub_graph_config;
  sub_graph_config.allow_taps = false;
  sub_graph_config.max_depth = max_depth;
  sub_graph_config.max_exits = 1;
  OnlyValidPartitionRule valid("", std::move(combine), sub_graph_config);
  // Wrap the candidates in a "Primitive" function with a "Compiler" attribute.
  return PrimitivePartitionRule("", std::move(valid));
}

TVM_REGISTER_GLOBAL("relay.collage.MakeLabelledDFPatternPartitionRule")
    .set_body_typed(MakeLabelledDFPatternPartitionRule);

TVM_REGISTER_GLOBAL("relay.collage.MakeLabelledDFPatternPartitionRuleWithPredicate")
    .set_body_typed(MakeLabelledDFPatternPartitionRule);

TVM_REGISTER_GLOBAL("relay.collage.MakePatternBYOCPartitionRule")
    .set_body_typed(MakePatternBYOCPartitionRule);

/*!
 * \brief Returns the rule to pick out expression nodes which can be 'left behind' for execution
 * on the host.
 */
PartitionRule MakeHostPartitionRule() { return HostPartitionRule(""); }

}  // namespace

Array<PartitionSpec> GatherPartitionSpecs(const CompilationConfig& config) {
  Array<PartitionSpec> result;
  for (const auto& primitive_target : config->primitive_targets) {
    String spec_name = GetSpecName(primitive_target);
    PartitionRule rule;
    if (primitive_target.IsExternalCodegen()) {
      // Transition to the Python side so we can get access to the BYOC pattern registry.
      // That will bounce right back into the above construction helpers.
      static const runtime::PackedFunc* make_byoc_partition_rule =
          runtime::Registry::Get("tvm.relay.collage.make_byoc_partition_rule");
      ICHECK(make_byoc_partition_rule);
      rule = (*make_byoc_partition_rule)(spec_name);  // spec_name == primitive_target->kind->name
      VLOG(1) << "Target " << primitive_target->ToDebugString() << " is for BYOC spec_name "
              << spec_name << " and has default partition rule:\n"
              << rule->ToString();
    } else {
      rule = MakeTVMPartitionRule();
      VLOG(1) << "Target " << primitive_target->ToDebugString() << " is for TVM spec_name "
              << spec_name << " and has default partition rule:\n"
              << rule->ToString();
    }
    result.push_back(PartitionSpec(spec_name, primitive_target, rule));
  }

  // Add one more spec to cover the host target.
  result.push_back(PartitionSpec(kHostSpecName, config->host_target, MakeHostPartitionRule()));

  return result;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
