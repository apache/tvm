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
#include "../../target/parsers/aprofile.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

String GetRuleKindFromTarget(const Target& target) {
  if (target->kind->name == "llvm") {
    static const PackedFunc* target_has_feature_fn_ptr =
        runtime::Registry::Get("target.target_has_feature");
    ICHECK(target_has_feature_fn_ptr != nullptr)
        << "The `target.target_has_feature` func is not in tvm registry.";
    bool have_avx512vnni = (*target_has_feature_fn_ptr)("avx512vnni", target);
    bool have_avxvnni = (*target_has_feature_fn_ptr)("avxvnni", target);
    if (have_avx512vnni || have_avxvnni) {
      return "vnni";
    } else {
      bool have_avx512f = (*target_has_feature_fn_ptr)("avx512f", target);
      bool have_avx512bw = (*target_has_feature_fn_ptr)("avx512bw", target);
      if (have_avx512bw && have_avx512f) {
        return "avx512";
      }
    }

    TargetJSON target_json = target::parsers::aprofile::ParseTarget(target->Export());
    TargetFeatures afeatures = Downcast<TargetFeatures>(target_json.at("features"));

    if (Downcast<Bool>(afeatures.at("has_dotprod"))) {
      return "dotprod";
    }
    if (Downcast<Bool>(afeatures.at("has_asimd"))) {
      return "asimd";
    }
    return "llvm";
  }
  if (target->kind->name == "hexagon") {
    return "hexagon";
  }
  if (target->kind->name == "cuda") {
    if (Optional<String> opt_sm = target->GetAttr<String>("arch")) {
      std::string sm = opt_sm.value();
      if (support::StartsWith(sm, "sm_")) {
        sm = sm.substr(3);
        try {
          if (std::stoi(sm) >= 75) {
            return "cuda-tensorcore";
          }
        } catch (const std::invalid_argument& e) {
          LOG(WARNING) << "ValueError: Unable to parse `target.arch`: " << sm
                       << ". Details: " << e.what();
        }
      }
    }
    return "cuda";
  }

  if (IsGPUTarget(target->kind->name)) {
    return "cuda";
  }

  if (target->kind->name == "c") {
    return "c";
  }
  LOG(FATAL) << "Unsupported target: " << target;
  throw;
}

void SpaceGeneratorNode::InitializeWithTuneContext(const TuneContext& context) {
  if (context->target.defined() &&  //
      !(sch_rules.defined() &&      //
        postprocs.defined() &&      //
        mutator_probs.defined())) {
    String kind = GetRuleKindFromTarget(context->target.value());
    Array<ScheduleRule> default_sch_rules;
    Array<Postproc> default_postprocs;
    Map<Mutator, FloatImm> default_mutator_probs;
    // for target with skylake-avx512
    if (kind == "llvm") {
      default_sch_rules = ScheduleRule::DefaultLLVM();
      default_postprocs = Postproc::DefaultLLVM();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "cuda") {
      default_sch_rules = ScheduleRule::DefaultCUDA();
      default_postprocs = Postproc::DefaultCUDA();
      default_mutator_probs = Mutator::DefaultCUDA();
    } else if (kind == "cuda-tensorcore") {
      default_sch_rules = ScheduleRule::DefaultCUDATensorCore();
      default_postprocs = Postproc::DefaultCUDATensorCore();
      default_mutator_probs = Mutator::DefaultCUDATensorCore();
    } else if (kind == "hexagon") {
      default_sch_rules = ScheduleRule::DefaultHexagon();
      default_postprocs = Postproc::DefaultHexagon();
      default_mutator_probs = Mutator::DefaultHexagon();
    } else if (kind == "vnni") {
      default_sch_rules = ScheduleRule::DefaultX86("vnni");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "avx512") {
      default_sch_rules = ScheduleRule::DefaultX86("avx512");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "asimd") {
      default_sch_rules = ScheduleRule::DefaultARM("neon");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else if (kind == "dotprod") {
      default_sch_rules = ScheduleRule::DefaultARM("dotprod");
      default_postprocs = Postproc::DefaultCPUTensorization();
      default_mutator_probs = Mutator::DefaultLLVM();
    } else {
      LOG(FATAL) << "Unsupported kind: " << kind;
      throw;
    }
    if (!sch_rules.defined()) {
      sch_rules = default_sch_rules;
    }
    if (!postprocs.defined()) {
      postprocs = default_postprocs;
    }
    if (!mutator_probs.defined()) {
      mutator_probs = default_mutator_probs;
    }
  }
  if (sch_rules.defined()) {
    for (ScheduleRule i : sch_rules.value()) {
      i->InitializeWithTuneContext(context);
    }
  }
  if (postprocs.defined()) {
    for (Postproc i : postprocs.value()) {
      i->InitializeWithTuneContext(context);
    }
  }
  if (mutator_probs.defined()) {
    for (const auto& kv : mutator_probs.value()) {
      Mutator mutator = kv.first;
      mutator->InitializeWithTuneContext(context);
    }
  }
}

void PySpaceGeneratorNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PySpaceGenerator's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

Array<tir::Schedule> PySpaceGeneratorNode::GenerateDesignSpace(const IRModule& mod) {
  ICHECK(f_generate_design_space != nullptr)
      << "PySpaceGenerator's GenerateDesignSpace method not implemented!";
  return f_generate_design_space(mod);
}

SpaceGenerator PySpaceGeneratorNode::Clone() const {
  ICHECK(f_clone != nullptr) << "PySpaceGenerator's Clone method not implemented!";
  return f_clone();
}

SpaceGenerator SpaceGenerator::PySpaceGenerator(
    Optional<Array<ScheduleRule>> sch_rules, Optional<Array<Postproc>> postprocs,
    Optional<Map<Mutator, FloatImm>> mutator_probs,
    FInitializeWithTuneContext f_initialize_with_tune_context,
    FGenerateDesignSpace f_generate_design_space, FClone f_clone) {
  ObjectPtr<PySpaceGeneratorNode> n = make_object<PySpaceGeneratorNode>();
  n->sch_rules = sch_rules;
  n->postprocs = postprocs;
  n->mutator_probs = mutator_probs;
  n->f_initialize_with_tune_context = std::move(f_initialize_with_tune_context);
  n->f_generate_design_space = std::move(f_generate_design_space);
  n->f_clone = std::move(f_clone);
  return SpaceGenerator(n);
}

TVM_REGISTER_OBJECT_TYPE(SpaceGeneratorNode);
TVM_REGISTER_NODE_TYPE(PySpaceGeneratorNode);

TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorInitializeWithTuneContext")
    .set_body_method<SpaceGenerator>(&SpaceGeneratorNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorGenerateDesignSpace")
    .set_body_method<SpaceGenerator>(&SpaceGeneratorNode::GenerateDesignSpace);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorPySpaceGenerator")
    .set_body_typed(SpaceGenerator::PySpaceGenerator);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorClone")
    .set_body_method<SpaceGenerator>(&SpaceGeneratorNode::Clone);

}  // namespace meta_schedule
}  // namespace tvm
