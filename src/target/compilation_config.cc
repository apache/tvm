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
  v->Visit("default_primitive_virtual_device", &default_primitive_virtual_device);
  v->Visit("host_virtual_device", &host_virtual_device);
  v->Visit("optional_homogenous_target", &optional_homogeneous_target);
  // NOTE: The virtual_device_cache_ is not accessible via FFI.
}

VirtualDevice CompilationConfigNode::CanonicalVirtualDevice(
    const VirtualDevice& virtual_device) const {
  if (virtual_device->target.defined()) {
    return virtual_device_cache_.Unique(virtual_device);
  }
  DLDeviceType device_type = virtual_device->device_type();
  // TODO(mbs): Proper diagnostics.
  CHECK(device_type != kInvalidDeviceType)
      << "VirtualDevice annotations must include at least a device_type";
  Target target = FindPrimitiveTargetOrFail(virtual_device->device_type());
  return virtual_device_cache_.Unique(VirtualDevice(device_type, virtual_device->virtual_device_id,
                                                    target, virtual_device->memory_scope));
}

void CompilationConfigNode::EstablishDefaultVirtualDevices(const transform::PassContext& pass_ctx) {
  //
  // Gather the hints as to what our default device type for the 'host' should be, and
  // create an appropriate target if we don't already have one.
  //
  DLDeviceType host_device_type;
  if (host_target.defined()) {
    CHECK(!host_target->host.defined()) << "Host targets are not expected to have hosts";
    host_device_type = static_cast<DLDeviceType>(host_target->kind->device_type);
    VLOG(1) << "Using the given host target " << host_target->ToDebugString() << " of device type "
            << host_device_type << " for the host target";
    for (const auto& primitive_target : primitive_targets) {
      if (primitive_target->host.defined() &&
          !StructuralEqual()(primitive_target->host, host_target)) {
        VLOG(1) << "The primitive target " << primitive_target->ToDebugString()
                << " already has a host which disagrees with the desired host target. It "
                << "will be ignored.";
      }
    }
  } else if (primitive_targets.size() == 1 && primitive_targets.front()->host.defined()) {
    host_target = primitive_targets.front()->GetHost().value();
    CHECK(!host_target->host.defined()) << "Host targets are not expected to have hosts";
    host_device_type = static_cast<DLDeviceType>(host_target->kind->device_type);
    VLOG(1) << "Using the host of the unique primitive target, namely "
            << host_target->ToDebugString() << " of device type " << host_device_type
            << " for the host target";
  } else if (primitive_targets.size() == 1 &&
             primitive_targets.front()->kind->device_type == kDLCPU) {
    // In the homogenous case without an explicit host target just use the given target so long as
    // it's a CPU.
    host_device_type = kDLCPU;
    host_target = primitive_targets.front();
    VLOG(1) << "Using the unique primitive target " << host_target->ToDebugString()
            << " of device type " << host_device_type << " for the host target";
  } else {
    // Fallback.
    host_device_type = kDLCPU;
    // Even if the list of available targets already includes one for kDLCPU we won't use it
    // in the hetrogeneous case since its options may not be appropriate for host code
    // (eg shape functions). Instead, create a fresh default Target.
    host_target = MakeDefaultTarget(host_device_type);
    VLOG(1) << "Using the default target " << host_target->ToDebugString() << " of device type "
            << host_device_type << " for the host target";
  }
  ICHECK(host_target.defined());
  ICHECK(!host_target->host.defined());

  if (host_device_type != kDLCPU) {
    // I think we're on thin ice here until we've audited the code base for assumed kDLCPU.
    VLOG(1) << "The host target is not a CPU.";
  }

  //
  // Establish the host VirtualDevice.
  //
  host_virtual_device =
      virtual_device_cache_.Unique(VirtualDevice(host_device_type,
                                                 /*virtual_device_id=*/0, host_target));

  //
  // Now that we've settled on a host, we can set it as the host on all primitive targets.
  //
  Array<Target> new_primitve_targets;
  new_primitve_targets.reserve(primitive_targets.size());
  for (const auto& primitive_target : primitive_targets) {
    new_primitve_targets.push_back(Target(primitive_target, host_target));
  }
  primitive_targets = new_primitve_targets;

  //
  // Gather the hints as to what our default device type for primitives should be.
  //
  DLDeviceType default_primitive_device_type;
  Optional<Integer> opt_fallback_dev = pass_ctx->GetConfig<Integer>("relay.fallback_device_type");
  if (opt_fallback_dev) {
    const int64_t v = opt_fallback_dev.value()->value;
    CHECK_GT(v, 0)
        << "The 'relay.fallback_device_type' pass attribute is set to an invalid device type " << v;
    default_primitive_device_type = static_cast<DLDeviceType>(v);
    VLOG(1) << "Using the 'relay.fallback_device_type' pass attribute "
            << default_primitive_device_type
            << " as the default device type for all primitive operations";
  } else if (primitive_targets.size() == 1) {
    // In the homogeneous case there's no free choice.
    default_primitive_device_type =
        static_cast<DLDeviceType>(primitive_targets.front()->kind->device_type);
    VLOG(1) << "Using the device type " << default_primitive_device_type
            << " of the unique primitive target as the default device type for all primitive "
            << "operations";
  } else {
    // Fallback. Note that we'll require a primitive Target of kDLCPU device_type to be given
    // and won't manufacture one out of thin air.
    default_primitive_device_type = kDLCPU;
    VLOG(1) << "Using " << default_primitive_device_type
            << " as the default device type for all primitive operations";
  }

  //
  // Establish the default primitive VirtualDevice, choosing a known Target to match the device
  // type.
  //
  default_primitive_virtual_device = virtual_device_cache_.Unique(VirtualDevice(
      default_primitive_device_type,
      /*virtual_device_id=*/0, FindPrimitiveTargetOrFail(default_primitive_device_type)));
}

/* static */ Target CompilationConfigNode::MakeDefaultTarget(DLDeviceType device_type) {
  std::string name = runtime::DeviceName(device_type);
  if (name == "cpu") {
    if (runtime::Registry::Get("codegen.LLVMModuleCreate")) {
      // LLVM is available.
      // TODO(mbs): More robust extension mechanism?
      return Target("llvm");
    } else {
      // LLVM is not available.
      // TODO(mbs): Already deprecated?
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
                                     TargetMap legacy_target_map_arg,
                                     Target optional_host_target_arg) {
  VLOG_CONTEXT << "CompilationConfig";

  auto node = make_object<CompilationConfigNode>();

  for (const auto& pair : legacy_target_map_arg) {
    VLOG(0) << "Available primitive target " << pair.first << " = " << pair.second->ToDebugString();
  }
  if (optional_host_target_arg.defined()) {
    VLOG(0) << "Available host target " << optional_host_target_arg->ToDebugString();
  }

  // Capture the arguments in our preferred representation.
  for (const auto& pair : legacy_target_map_arg) {
    node->primitive_targets.push_back(pair.second);
  }
  node->host_target = optional_host_target_arg;

  // Complete the targets vector and establish default scopes. After this primitive_targets will
  // contain the definitive list of all required targets, target_host will be defined, and
  // all primitive targets will have host target_host.
  node->EstablishDefaultVirtualDevices(pass_ctx);

  // LEGACY: Reconstruct the target map from all the primitive targets.
  // Note that we require pointer equality between targets in legacy_target_map and
  // primitive_targets.
  for (const auto& primitive_target : node->primitive_targets) {
    node->legacy_target_map.Set(Integer(primitive_target->kind->device_type), primitive_target);
  }

  ICHECK(node->default_primitive_virtual_device->target.defined());
  ICHECK(node->host_virtual_device->target.defined());
  ICHECK_GT(node->primitive_targets.size(), 0U);

  // Legacy: Some passes only support homogenous compilation and expect the target to be
  // given by the global target context. Make this easy to detect.
  node->optional_homogeneous_target =
      node->legacy_target_map.size() == 1 ? (*node->legacy_target_map.begin()).second : Target();

  for (const auto& target : node->primitive_targets) {
    VLOG(1) << "Target " << target->ToDebugString() << " of device type "
            << target->kind->device_type << " is available for primitives";
  }
  VLOG(1) << "Using default primitive virtual device " << node->default_primitive_virtual_device;
  VLOG(1) << "Using host virtual device " << node->host_virtual_device;

  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("target.MakeCompilationConfig")
    .set_body_typed([](const transform::PassContext& pass_ctx, TargetMap legacy_target_map,
                       Target optional_host_target) -> CompilationConfig {
      return CompilationConfig(pass_ctx, std::move(legacy_target_map),
                               std::move(optional_host_target));
    });

}  // namespace tvm
