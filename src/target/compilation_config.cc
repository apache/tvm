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
  v->Visit("host_target", &host_target);
  v->Visit("primitive_targets", &primitive_targets);
  v->Visit("default_primitive_virtual_device", &default_primitive_virtual_device);
  v->Visit("host_virtual_device", &host_virtual_device);
  v->Visit("optional_homogenous_target", &optional_homogeneous_target);
  // NOTE: The virtual_device_cache_ is not accessible via FFI.
}

Target CompilationConfigNode::FindPrimitiveTargetForDeviceOrFail(DLDeviceType device_type) const {
  ICHECK_GT(device_type, 0) << "Invalid device type";
  auto itr = std::find_if(
      primitive_targets.begin(), primitive_targets.end(),
      [device_type](const Target& target) { return target->GetTargetDeviceType() == device_type; });
  if (itr == primitive_targets.end()) {
    std::stringstream msg;
    msg << "No target is specified for device type " << device_type
        << ". The available device types and targets are:" << std::endl;
    for (const auto& target : primitive_targets) {
      msg << "  " << target->GetTargetDeviceType() << "-> " << target->ToDebugString() << std::endl;
    }
    LOG(FATAL) << msg.str();
  }
  return *itr;
}

Optional<Target> CompilationConfigNode::FindPrimitiveTargetForKind(
    const std::string& kind_name) const {
  Optional<TargetKind> opt_kind = TargetKind::Get(kind_name);
  if (!opt_kind.defined()) {
    VLOG(1) << "No such target kind for '" << kind_name << "'";
    return {};
  }
  auto itr =
      std::find_if(primitive_targets.begin(), primitive_targets.end(),
                   [kind_name](const Target& target) { return target->kind->name == kind_name; });
  if (itr == primitive_targets.end()) {
    VLOG(1) << "No target available matching kind '" << kind_name << "'";
    return {};
  }
  return *itr;
}

Target CompilationConfigNode::CanonicalTarget(const Target& target) const {
  // Fast path -- object identity.
  if (target == host_target) {
    return target;
  }
  for (const auto& primitive_target : primitive_targets) {
    if (target == primitive_target) {
      return target;
    }
  }
  // Slow path -- structural equality. We have so few targets it does not seem worth building an
  // index.
  if (StructuralEqual()(target, host_target)) {
    return host_target;
  }
  for (const auto& primitive_target : primitive_targets) {
    if (StructuralEqual()(target, primitive_target)) {
      return primitive_target;
    }
  }
  // No match.
  return target;
}

VirtualDevice CompilationConfigNode::CanonicalVirtualDevice(
    const VirtualDevice& virtual_device) const {
  // Targets need special handling.
  Target target = virtual_device->target;
  if (target.defined()) {
    // It is possible the given target object was constructed by the user, but was then
    // rewritten on the way into the CompilationConfig. So 'canonicalize' it by replacing
    // the given target with one structurally equal to one already known in the config if
    // possible.
    Target canon_target = CanonicalTarget(target);
    if (canon_target != target) {
      VLOG(1) << "Canonicalized target " << canon_target->ToDebugString();
    }
    target = canon_target;
  } else if (virtual_device->device_type() != kInvalidDeviceType) {
    // Since no target was given, choose one with a matching device type.
    // This is the one place where we allow device types to imply targets.
    target = FindPrimitiveTargetForDeviceOrFail(virtual_device->device_type());
    VLOG(1) << "Defaulted to target " << target->ToDebugString();
  }
  // else: the target will remain unknown.

  // Redirect to an existing structurally equal virtual device.
  return virtual_device_cache_.Unique(VirtualDevice(virtual_device->device_type(),
                                                    virtual_device->virtual_device_id, target,
                                                    virtual_device->memory_scope));
}

void CompilationConfigNode::Init(const transform::PassContext& pass_ctx,
                                 const Array<Target>& raw_targets) {
  VLOG_CONTEXT << "CompilationConfig";
  CHECK_GT(raw_targets.size(), 0U) << "Require at least one target";

  //
  // Decide on the host target.
  //

  // Any targets which could act as a host?
  auto hosting_itr = std::find_if(raw_targets.begin(), raw_targets.end(), [](const Target& target) {
    // TODO(tvm-team): The kDLHexagon device can act as a host. We can remove kDLHexagon
    // here once we refactored kDLHexagon to kDLCPU.
    return target->GetTargetDeviceType() == kDLCPU || target->GetTargetDeviceType() == kDLHexagon;
  });

  // Any targets with their host field set?
  auto has_host_itr = std::find_if(raw_targets.begin(), raw_targets.end(),
                                   [](const Target& target) { return target->host.defined(); });

  if (has_host_itr != raw_targets.end()) {
    // RULE A: If any raw target has a host, use the first such host for all the primitive
    // targets.
    host_target = Target((*has_host_itr)->GetHost().value(), /*host=*/Target());
    VLOG(1) << "The target " << (*has_host_itr)->ToDebugString() << " supplies a host target "
            << host_target->ToDebugString() << " of device type "
            << host_target->GetTargetDeviceType();
  } else if (hosting_itr != raw_targets.end()) {
    // RULE B: If any raw target is for a device which could be a host then use the first such as
    // the host.
    host_target = Target(*hosting_itr, /*host=*/Target());
    VLOG(1) << "Using target " << host_target->ToDebugString() << " of CPU-like device type "
            << host_target->GetTargetDeviceType() << " as the host target";
  } else {
    // RULE C: Otherwise, create a default CPU host target.
    host_target = MakeDefaultCPUTarget();
    VLOG(1) << "Created a default target " << host_target->ToDebugString() << " of device type "
            << host_target->GetTargetDeviceType() << " for the host target";
  }
  ICHECK(host_target.defined());
  ICHECK(!host_target->host.defined());

  if (host_target->GetTargetDeviceType() != kDLCPU) {
    // I think we're on thin ice here until we've audited the code base for assumed CPU hosts.
    VLOG(1) << "The host target is not a CPU. This is probably not going to work.";
  }

  //
  // Establish the host VirtualDevice.
  //
  host_virtual_device = virtual_device_cache_.Unique(
      VirtualDevice(static_cast<DLDeviceType>(host_target->GetTargetDeviceType()),
                    /*virtual_device_id=*/0, host_target));
  ICHECK(host_virtual_device.defined());
  ICHECK(host_virtual_device->target.defined());

  //
  // Now that we've settled on a host, we can set it as the host on all the raw targets.
  //
  primitive_targets.clear();
  primitive_targets.reserve(raw_targets.size());
  for (const auto& raw_target : raw_targets) {
    if (raw_target->host.defined() && !StructuralEqual()(raw_target->host, host_target)) {
      VLOG(1) << "The target " << raw_target->ToDebugString()
              << " already has a host which disagrees with the desired host target. It "
              << "will be overridden.";
    }
    primitive_targets.push_back(Target(raw_target, host_target));
  }
  ICHECK_GT(primitive_targets.size(), 0U);

  //
  // Check the primitive_targets are ordered correctly re Target::IsExternalCodegenFor,
  // and make sure no two targets share a kind name.
  //

  // TODO(mbs): We could just sort the list, but given all the implicit defaulting for backwards
  // compat it seems we should avoid making this any more magical than necessary. But revisit
  // if usability suffers.
  std::unordered_set<DLDeviceType> primitive_target_device_types;
  std::unordered_set<std::string> kind_names;
  for (const auto& target : primitive_targets) {
    primitive_target_device_types.emplace(static_cast<DLDeviceType>(target->GetTargetDeviceType()));
    CHECK(kind_names.emplace(target->kind->name).second) << "Multiple targets have been given"
                                                            "for the same device kind '"
                                                         << target->kind->name << "'";
  }
  for (DLDeviceType device_type : primitive_target_device_types) {
    Target first_primitive_target;
    for (const auto& current_primitive_target : primitive_targets) {
      if (current_primitive_target->GetTargetDeviceType() != device_type) {
        continue;
      }
      if (!first_primitive_target.defined()) {
        first_primitive_target = current_primitive_target;
        // Note it is valid to have only one external codegen target.
      } else {
        CHECK(current_primitive_target.IsExternalCodegenFor(first_primitive_target))
            << "When given multiple targets for the device type " << device_type
            << " the first must be for non external codegen, and all subsequent must be for "
               "external codegen. However have been given first "
            << first_primitive_target->ToDebugString() << " and subsequent "
            << current_primitive_target->ToDebugString();
      }
    }
  }

  //
  // Decide on the default device type for primitives.
  //
  DLDeviceType default_primitive_device_type;
  Optional<Integer> opt_fallback_dev = pass_ctx->GetConfig<Integer>("relay.fallback_device_type");
  if (opt_fallback_dev) {
    // RULE D: Respect the PassContext setting if given.
    const int64_t v = opt_fallback_dev.value()->value;
    CHECK_GT(v, 0)
        << "The 'relay.fallback_device_type' pass attribute is set to an invalid device type " << v;
    default_primitive_device_type = static_cast<DLDeviceType>(v);
    VLOG(1) << "Using the 'relay.fallback_device_type' pass attribute "
            << default_primitive_device_type
            << " as the default device type for all primitive operations";
  } else if (primitive_target_device_types.size() == 1) {
    // RULE E: Since only one device in use there's no choice to make.
    default_primitive_device_type = *primitive_target_device_types.begin();
    VLOG(1) << "All primitive targets have the device type " << default_primitive_device_type
            << " so that is also the default device type for all primitive operations.";
  } else {
    // RULE F: Fallback to CPU.
    default_primitive_device_type = kDLCPU;
    VLOG(1) << "Using " << default_primitive_device_type
            << " as the default device type for all primitive operations";
  }

  //
  // Establish the default primitive VirtualDevice, choosing a known Target to match the device
  // type. We do not create a default target, it must already exist as a primitive target.
  //
  default_primitive_virtual_device = CanonicalVirtualDevice(
      VirtualDevice::ForDeviceType(default_primitive_device_type, /*virtual_device_id=*/0));

  ICHECK(default_primitive_virtual_device.defined());
  ICHECK(default_primitive_virtual_device->target.defined());

  // Legacy: Some passes only support homogenous compilation and expect the target to be
  // given by the global target context. Make this easy to detect.
  optional_homogeneous_target =
      primitive_targets.size() == 1 ? *primitive_targets.begin() : Target();
}

/* static */ Target CompilationConfigNode::MakeDefaultCPUTarget() {
  if (runtime::Registry::Get("codegen.LLVMModuleCreate")) {
    // LLVM is available.
    // TODO(mbs): More robust extension mechanism?
    return Target("llvm");
  } else {
    // LLVM is not available.
    // TODO(mbs): Already deprecated?
    return Target("stackvm");
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CompilationConfigNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = ref.as<CompilationConfigNode>();
      p->stream << "Primitive targets:";
      for (const auto& target : node->primitive_targets) {
        p->stream << std::endl
                  << "  " << target->GetTargetDeviceType() << " |-> " << target->ToDebugString();
      }
      p->stream << std::endl
                << "Default primitive virtual device: " << node->default_primitive_virtual_device;
      p->stream << std::endl << "Host virtual device: " << node->host_virtual_device;
    });

CompilationConfig::CompilationConfig(const transform::PassContext& pass_ctx,
                                     const Array<Target>& raw_targets) {
  auto node = make_object<CompilationConfigNode>();
  node->Init(pass_ctx, raw_targets);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("target.MakeCompilationConfig")
    .set_body_typed([](const transform::PassContext& pass_ctx,
                       const Array<Target>& raw_targets) -> CompilationConfig {
      return CompilationConfig(pass_ctx, raw_targets);
    });

}  // namespace tvm
