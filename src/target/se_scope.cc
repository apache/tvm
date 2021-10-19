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
 * \file tvm/target/se_scope.cc
 * \brief Implementation of \p SEScope for representing a Storage or Execution scope.
 */
#include <tvm/node/reflection.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/se_scope.h>

namespace tvm {

/*! \brief A cache of \p SEScopes. */
class SEScopeCache {
 public:
  SEScope MakeSEScope(DLDeviceType device_type, int virtual_device_id, Target target,
                      String memory_scope) {
    // Not the most efficient, but reducing the key to a string seems to be the simplest.
    // Note this means we are effectively collapsing Targets by their str() representation.
    std::ostringstream os;
    os << device_type;
    os << ":" << virtual_device_id;
    if (target.defined()) {
      os << ":'" << target->str() << "'";
    } else {
      os << ":null";
    }
    os << ":'" << memory_scope << "'";
    std::string key = os.str();
    auto itr = cache_.find(key);
    if (itr != cache_.end()) {
      return itr->second;
    }
    SEScope scope(device_type, virtual_device_id, std::move(target), std::move(memory_scope));
    cache_.emplace(key, scope);
    VLOG(1) << "new scope \"" << key << "\" -> " << scope;
    return scope;
  }

 private:
  std::unordered_map<std::string, SEScope> cache_;
};

/*! \brief Thread local cache of already constructed \p SEScopes. */
using ThreadLocalSEScopeCache = dmlc::ThreadLocalStore<SEScopeCache>;

TVM_REGISTER_NODE_TYPE(SEScopeNode);

void SEScopeNode::VisitAttrs(AttrVisitor* v) {
  int i = static_cast<int>(device_type_);
  v->Visit("device_type", &i);
  device_type_ = static_cast<DLDeviceType>(i);
  v->Visit("virtual_device_id", &virtual_device_id_);
  v->Visit("target", &target_);
  v->Visit("memory_scope", &memory_scope_);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SEScopeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = ref.as<SEScopeNode>();
      p->stream << "SEScopeNode(";
      if (node->is_fully_unconstrained()) {
        p->stream << "?";
      } else {
        bool need_sep = false;
        if (node->device_type() != kInvalidDeviceType) {
          p->stream << "device_type=" << node->device_type();
          need_sep = true;
        }
        if (node->virtual_device_id() >= 0) {
          if (need_sep) {
            p->stream << ", ";
          }
          p->stream << "virtual_device_id=" << node->virtual_device_id();
          need_sep = true;
        }
        if (node->target().defined()) {
          if (need_sep) {
            p->stream << ", ";
          }
          p->stream << "target='" << node->target()->str() << "'";
          need_sep = true;
        }
        if (!node->memory_scope().empty()) {
          if (need_sep) {
            p->stream << ", ";
          }
          p->stream << "memory_scope='" << node->memory_scope() << "'";
        }
      }
      p->stream << ")";
    });

SEScope::SEScope(DLDeviceType device_type, int virtual_device_id, Target target,
                 String memory_scope) {
  ICHECK(!target.defined() || device_type == target->kind->device_type)
      << "target '" << target->str() << "' has device type " << target->kind->device_type
      << " but scope has device type " << device_type;
  auto object = make_object<SEScopeNode>();
  object->device_type_ = device_type;
  object->virtual_device_id_ = virtual_device_id;
  object->target_ = std::move(target);
  object->memory_scope_ = std::move(memory_scope);
  data_ = std::move(object);
}

/* static */
SEScope SEScope::MakeSEScope(DLDeviceType device_type, int virtual_device_id, Target target,
                             String memory_scope) {
  return ThreadLocalSEScopeCache ::Get()->MakeSEScope(device_type, virtual_device_id,
                                                      std::move(target), std::move(memory_scope));
}

/* static */
Optional<SEScope> SEScope::Join(const SEScope& lhs, const SEScope& rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  DLDeviceType joined_device_type;
  if (lhs->device_type_ != kInvalidDeviceType) {
    joined_device_type = lhs->device_type_;
    if (rhs->device_type_ != kInvalidDeviceType && lhs->device_type_ != rhs->device_type_) {
      return {};
    }
  } else {
    joined_device_type = rhs->device_type_;
  }
  int joined_virtual_device_id;
  if (lhs->virtual_device_id_ >= 0) {
    joined_virtual_device_id = lhs->virtual_device_id_;
    if (rhs->virtual_device_id_ >= 0 && lhs->virtual_device_id_ != rhs->virtual_device_id_) {
      return {};
    }
  } else {
    joined_virtual_device_id = rhs->virtual_device_id_;
  }
  Target joined_target;
  if (lhs->target_.defined()) {
    joined_target = lhs->target_;
    if (rhs->target_.defined() && lhs->target_ != rhs->target_) {
      return {};
    }
  } else {
    joined_target = rhs->target_;
  }
  String joined_memory_scope;
  if (!lhs->memory_scope_.empty()) {
    joined_memory_scope = lhs->memory_scope_;
    if (!rhs->memory_scope_.empty() && lhs->memory_scope_ != rhs->memory_scope_) {
      return {};
    }
  } else {
    joined_memory_scope = rhs->memory_scope_;
  }
  return MakeSEScope(joined_device_type, joined_virtual_device_id, joined_target,
                     joined_memory_scope);
}

/* static */
SEScope SEScope::Default(const SEScope& lhs, const SEScope& rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  DLDeviceType defaulted_device_type;
  if (lhs->device_type_ != kInvalidDeviceType) {
    defaulted_device_type = lhs->device_type_;
  } else {
    defaulted_device_type = rhs->device_type_;
  }
  int defaulted_virtual_device_id;
  if (lhs->virtual_device_id_ >= 0) {
    defaulted_virtual_device_id = lhs->virtual_device_id_;
  } else {
    defaulted_virtual_device_id = rhs->virtual_device_id_;
  }
  Target defaulted_target;
  if (lhs->target_.defined()) {
    defaulted_target = lhs->target_;
  } else {
    // We can only default to the rhs's target if it is consistent with the device type
    if (rhs->target_.defined() && rhs->target_->kind->device_type == defaulted_device_type) {
      defaulted_target = rhs->target_;
    }
    // else: leave as null
  }
  String defaulted_memory_scope;
  if (!lhs->memory_scope_.empty()) {
    defaulted_memory_scope = lhs->memory_scope_;
  } else {
    defaulted_memory_scope = rhs->memory_scope_;
  }
  return MakeSEScope(defaulted_device_type, defaulted_virtual_device_id, defaulted_target,
                     defaulted_memory_scope);
}

namespace {
/*!
 * \brief Returns a freshly constructed \p Target to represent \p device_type.
 */
Target MakeDefaultTarget(DLDeviceType device_type) {
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

/*!
 * \brief Return the \p Target to use for \p device_type, possibly by stealing the \p host_target,
 * or by creating a fresh target.
 */
Target FindOrAddDefault(Array<Target>* targets, const Target& optional_host_target,
                        DLDeviceType device_type) {
  auto itr = std::find_if(targets->begin(), targets->end(), [device_type](const Target& target) {
    return target->kind->device_type == device_type;
  });
  if (itr == targets->end()) {
    if (optional_host_target.defined() && optional_host_target->kind->device_type == device_type) {
      LOG(INFO) << "Using the given host target '" << optional_host_target->str()
                << "' for device type " << device_type;
      targets->push_back(optional_host_target);
      return optional_host_target;
    } else {
      Target target = MakeDefaultTarget(device_type);
      LOG(WARNING) << "No target has been given for the device type " << device_type
                   << " in the targets list. Creating a default target '" << target->str()
                   << "' for that device";
      targets->push_back(target);
      return target;
    }
  } else {
    return *itr;
  }
}

/*!
 * \brief Returns the default \p SEScope for primitives and the \p SEScope for the host
 * given vector of available \p targets. If necessary, add new \p Targets to \p targets
 * to match the required devices.
 */
std::pair<SEScope, SEScope> EstablishDefaultSEScopes(const transform::PassContext& pass_ctx,
                                                     Array<Target>* targets,
                                                     const Target& optional_host_target) {
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
  } else if (targets->size() == 1) {
    // In the homogeneous case there's no free choice.
    default_primitive_device_type = static_cast<DLDeviceType>(targets->front()->kind->device_type);
    LOG(INFO) << "Using the unique target '" << targets->front()->str() << "' of device type "
              << default_primitive_device_type
              << " as the default device type for all primitive operations";
  } else {
    default_primitive_device_type = kDLCPU;
    LOG(WARNING) << "Using " << default_primitive_device_type
                 << " as the default device type for all primitive operations";
  }

  //
  // Gather the hints as to what our default device type for the 'host' should be.
  //
  DLDeviceType host_device_type;
  if (optional_host_target.defined()) {
    host_device_type = static_cast<DLDeviceType>(optional_host_target->kind->device_type);
    if (host_device_type != kDLCPU) {
      LOG(WARNING) << "Using the host target '" << optional_host_target->str()
                   << "' of non-CPU device type " << host_device_type
                   << " for all host operations and data";
    } else {
      LOG(INFO) << "Using the host target '" << optional_host_target->str() << "' of device type "
                << host_device_type << " for all host operations and data";
    }
  } else {
    host_device_type = kDLCPU;
    LOG(INFO) << "Using " << host_device_type
              << " as the device type for all host operations and data";
  }

  //
  // Now establish default targets
  //
  Target default_primitive_target =
      FindOrAddDefault(targets, optional_host_target, default_primitive_device_type);
  Target actual_host_target =
      optional_host_target.defined()
          ? optional_host_target
          : FindOrAddDefault(targets, optional_host_target, host_device_type);

  return {SEScope::MakeSEScope(default_primitive_device_type,
                               /*virtual_device_id=*/0, default_primitive_target),
          SEScope::MakeSEScope(host_device_type,
                               /*virtual_device_id=*/0, actual_host_target)};
}
}  // namespace

CompilationConfig::CompilationConfig(const transform::PassContext& pass_ctx,
                                     TargetMap legacy_target_map_arg,
                                     Target optional_host_target_arg)
    : legacy_target_map(std::move(legacy_target_map_arg)),
      optional_host_target(std::move(optional_host_target_arg)) {
  VLOG_CONTEXT << "CompilationConfig";
  for (const auto& pair : legacy_target_map) {
    VLOG(0) << "Available target " << pair.first << " = '" << pair.second->str() << "'";
  }
  if (optional_host_target.defined()) {
    VLOG(0) << "Available host target '" << optional_host_target->str() << "'";
  }

  // Legacy: Host & primitive targets need to be consistent.
  CheckAndUpdateHostConsistency(&legacy_target_map, &optional_host_target);

  // Gather the primitive targets as an ordinary vector.
  for (const auto& pair : legacy_target_map) {
    targets.push_back(pair.second);
  }

  // Complete the targets vector and establish default scopes. After this targets_ will contain
  // the definitive list of all required targets, both for host and primitives.
  auto pair = EstablishDefaultSEScopes(pass_ctx, &targets, optional_host_target);
  default_primitive_se_scope = pair.first;
  host_se_scope = pair.second;

  ICHECK(default_primitive_se_scope->target().defined());
  ICHECK(host_se_scope->target().defined());
  ICHECK_GT(targets.size(), 0U);

  // If we added a target to targets_ for the default primitive scope then we need to do the same in
  // the legacy target map. Note that we don't do the same for the host since the legacy map
  // is only supposed to track the targets for primitives. I think. Also note that TargetMap is
  // indexed by the *object identity* of the Integers for the device types so conveys nothing
  // beyond just vector of targets.
  auto itr = std::find_if(legacy_target_map.begin(), legacy_target_map.end(),
                          [this](const std::pair<Integer, Target>& pair) {
                            return pair.second->kind->device_type ==
                                   default_primitive_se_scope->device_type();
                          });
  if (itr == legacy_target_map.end()) {
    legacy_target_map.Set(static_cast<int>(default_primitive_se_scope->device_type()),
                          default_primitive_se_scope->target());
  }

  // Legacy: Some passes only support homogenous compilation and expect the target to be
  // given by the global target context.
  homogeneous_target =
      legacy_target_map.size() == 1 ? (*legacy_target_map.begin()).second : Target();

  for (const auto& target : targets) {
    VLOG(0) << "Established build target " << target->kind->device_type << " = '" << target->str()
            << "'";
  }
  VLOG(0) << "Established default primitive SEScope " << default_primitive_se_scope;
  VLOG(0) << "Established host SEScope " << host_se_scope;
}

TVM_REGISTER_GLOBAL("target.SEScope_ForDeviceTargetAndMemoryScope")
    .set_body_typed(SEScope::ForDeviceTargetAndMemoryScope);

}  // namespace tvm
