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
 * \file tvm/target/compilation_config.cc
 * \brief Implementation of \p CompilationConfig for collecting \p Targets.
 */

#include <tvm/runtime/device_api.h>
#include <tvm/target/compilation_config.h>

namespace tvm {

TVM_REGISTER_NODE_TYPE(CompilationConfigNode);

void CompilationConfigNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("legacy_target_map", &legacy_target_map);
  v->Visit("host_target", &host_target);
  v->Visit("primitive_targets", &primitive_targets);
  v->Visit("default_primitive_se_scope", &default_primitive_se_scope);
  v->Visit("host_se_scope", &host_se_scope);
  v->Visit("optional_homogenous_target", &optional_homogeneous_target);
  // NOTE: The se_scope_cache_ is not accessible via FFI.
}

SEScope CompilationConfigNode::CanonicalSEScope(const SEScope& se_scope) const {
  if (se_scope->target.defined()) {
    return se_scope_cache_.Unique(se_scope);
  }
  DLDeviceType device_type = se_scope->device_type();
  // TODO(mbs): Proper diagnostics.
  CHECK(device_type != kInvalidDeviceType)
      << "SEScope annotations must include at least a device_type";
  Target target = FindPrimitiveTargetOrFail(se_scope->device_type());
  return se_scope_cache_.Unique(
      SEScope(device_type, se_scope->virtual_device_id, target, se_scope->memory_scope));
}

void CompilationConfigNode::EstablishDefaultSEScopes(const transform::PassContext& pass_ctx) {
  //
  // Gather the hints as to what our default device type for primitives should be.
  //
  DLDeviceType default_primitive_device_type;
  Optional<Integer> opt_fallback_dev = pass_ctx->GetConfig<Integer>("relay.fallback_device_type");
  if (opt_fallback_dev) {
    const int64_t v = opt_fallback_dev.value()->value;
    if (v <= 0) {
      LOG(FATAL)
          << "The 'relay.fallback_device_type' pass attribute is set to an invalid device type "
          << v;
      default_primitive_device_type = kDLCPU;
    } else {
      default_primitive_device_type = static_cast<DLDeviceType>(v);
      LOG(INFO) << "Using the 'relay.fallback_device_type' pass attribute "
                << default_primitive_device_type
                << " as the default device type for all primitive operations";
    }
  } else if (primitive_targets.size() == 1) {
    // In the homogeneous case there's no free choice.
    default_primitive_device_type =
        static_cast<DLDeviceType>(primitive_targets.front()->kind->device_type);
    LOG(INFO) << "Using the unique target '" << primitive_targets.front()->str()
              << "' of device type " << default_primitive_device_type
              << " as the default device type for all primitive operations";
  } else {
    // Fallback. Note that we'll require a primitive Target of kDLCPU device_type to be given
    // and won't manufacture one out of thin air.
    default_primitive_device_type = kDLCPU;
    LOG(WARNING) << "Using " << default_primitive_device_type
                 << " as the default device type for all primitive operations";
  }

  //
  // Establish the default primitive SEScope, choosing a known Target to match the device type.
  //
  default_primitive_se_scope = se_scope_cache_.Unique(
      SEScope(default_primitive_device_type,
              /*virtual_device_id=*/0, FindPrimitiveTargetOrFail(default_primitive_device_type)));

  //
  // Gather the hints as to what our default device type for the 'host' should be.
  //
  DLDeviceType host_device_type;
  if (host_target.defined()) {
    host_device_type = static_cast<DLDeviceType>(host_target->kind->device_type);
    if (host_device_type != kDLCPU) {
      LOG(WARNING) << "Using the given host target '" << host_target->str()
                   << "' of non-CPU device type " << host_device_type
                   << " for all host operations and data";
    } else {
      LOG(INFO) << "Using the given host target '" << host_target->str() << "' of device type "
                << host_device_type << " for all host operations and data";
    }
  } else if (primitive_targets.size() == 1 &&
             primitive_targets.front()->kind->device_type == kDLCPU) {
    // In the homogenous case without an explicit host target just use the given target so long as
    // it's a CPU.
    host_device_type = kDLCPU;
    host_target =
        FindPrimitiveTargetOrFail(host_device_type);  // ie just primitive_targets.front()!
    LOG(INFO) << "Using the unique target '" << host_target->str() << "' of device type "
              << host_device_type << " for all host operations and data";
  } else {
    // Fallback.
    host_device_type = kDLCPU;
    // Even if the list of available targets already includes one for kDLCPU we won't use it
    // since its options may not be appropriate for host code (eg shape functions). Instead,
    // create a fresh default Target.
    host_target = MakeDefaultTarget(host_device_type);
    LOG(WARNING) << "Using the default host target '" << host_target->str() << "' of device type "
                 << host_device_type << " for all host operations and data";
  }

  //
  // Establish the host SEScope.
  //
  host_se_scope = se_scope_cache_.Unique(SEScope(host_device_type,
                                                 /*virtual_device_id=*/0, host_target));
}

/* static */ Target CompilationConfigNode::MakeDefaultTarget(DLDeviceType device_type) {
  std::string name = runtime::DeviceName(device_type);
  if (name == "cpu") {
    if (runtime::Registry::Get("codegen.LLVMModuleCreate")) {
      // LLVM is available.
      return Target("llvm");
    } else {
      // LLVM is not available.
      return Target("stackvm");
    }
  } else {
    return Target(name);
  }
}

Target CompilationConfigNode::FindPrimitiveTargetOrFail(DLDeviceType device_type) const {
  auto itr = std::find_if(
      primitive_targets.begin(), primitive_targets.end(),
      [device_type](const Target& target) { return target->kind->device_type == device_type; });
  CHECK(itr != primitive_targets.end()) << "No target for device type " << device_type << " in the "
                                        << primitive_targets.size() << " given by the targets list";
  return *itr;
}

CompilationConfig::CompilationConfig(const transform::PassContext& pass_ctx,
                                     TargetMap legacy_target_map_x, Target optional_host_target_x) {
  VLOG_CONTEXT << "CompilationConfig";

  auto node = make_object<CompilationConfigNode>();

  node->legacy_target_map = std::move(legacy_target_map_x);
  node->host_target = std::move(optional_host_target_x);

  for (const auto& pair : node->legacy_target_map) {
    VLOG(0) << "Available primitive target " << pair.first << " = '" << pair.second->str() << "'";
  }
  if (node->host_target.defined()) {
    VLOG(0) << "Available host target '" << node->host_target->str() << "'";
  }

  // Legacy: Make sure each primitive target host resolves to the given host target (if any).
  CheckAndUpdateHostConsistency(&node->legacy_target_map, &node->host_target);

  // Gather the primitive targets as an ordinary vector.
  for (const auto& pair : node->legacy_target_map) {
    node->primitive_targets.push_back(pair.second);
  }

  // Complete the targets vector and establish default scopes. After this targets_ will contain
  // the definitive list of all required targets, both for host and primitives.
  node->EstablishDefaultSEScopes(pass_ctx);

  ICHECK(node->default_primitive_se_scope->target.defined());
  ICHECK(node->host_se_scope->target.defined());
  ICHECK_GT(node->primitive_targets.size(), 0U);

  // Legacy: Some passes only support homogenous compilation and expect the target to be
  // given by the global target context.
  node->optional_homogeneous_target =
      node->primitive_targets.size() == 1 ? *node->primitive_targets.begin() : Target();

  for (const auto& target : node->primitive_targets) {
    LOG(INFO) << "Target '" << target->str() << "' of device type " << target->kind->device_type
              << " is available for primitives";
  }
  LOG(INFO) << "Using default primitive scope " << node->default_primitive_se_scope;
  LOG(INFO) << "Using host scope " << node->host_se_scope;

  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("target.MakeCompilationConfig")
    .set_body_typed([](const transform::PassContext& pass_ctx, TargetMap legacy_target_map,
                       Target optional_host_target) -> CompilationConfig {
      return CompilationConfig(pass_ctx, std::move(legacy_target_map),
                               std::move(optional_host_target));
    });

}  // namespace tvm
