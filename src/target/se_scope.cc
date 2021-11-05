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

TVM_REGISTER_NODE_TYPE(SEScopeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SEScopeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = ref.as<SEScopeNode>();
      p->stream << "SEScope(";
      if (node->IsFullyUnconstrained()) {
        p->stream << "?";
      } else {
        bool need_sep = false;
        if (node->device_type() != kInvalidDeviceType) {
          p->stream << "device_type=" << node->device_type();
          need_sep = true;
        }
        if (node->virtual_device_id >= 0) {
          if (need_sep) {
            p->stream << ", ";
          }
          p->stream << "virtual_device_id=" << node->virtual_device_id;
          need_sep = true;
        }
        if (node->target.defined()) {
          if (need_sep) {
            p->stream << ", ";
          }
          p->stream << "target=" << node->target->ToDebugString();
          need_sep = true;
        }
        if (!node->memory_scope.empty()) {
          if (need_sep) {
            p->stream << ", ";
          }
          p->stream << "memory_scope='" << node->memory_scope << "'";
        }
      }
#if TVM_LOG_DEBUG
      // We rely on object identity of SEScopes, so include the object address to help debugging.
      p->stream << ", id=" << reinterpret_cast<uint64_t>(ref.get());
#endif
      p->stream << ")";
    });

SEScope::SEScope(DLDeviceType device_type, int virtual_device_id, Target target,
                 MemoryScope memory_scope) {
  ICHECK(!target.defined() || device_type == target->kind->device_type)
      << "target " << target->ToDebugString() << " has device type " << target->kind->device_type
      << " but scope has device type " << device_type;
  auto node = make_object<SEScopeNode>();
  node->device_type_int = device_type;
  node->virtual_device_id = virtual_device_id;
  node->target = std::move(target);
  node->memory_scope = std::move(memory_scope);
  data_ = std::move(node);
}

/* static */ SEScope SEScope::FullyUnconstrained() {
  static const SEScope unconstrained{};
  return unconstrained;
}

/* static */
Optional<SEScope> SEScope::Join(const SEScope& lhs, const SEScope& rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  DLDeviceType joined_device_type;
  if (lhs->device_type() != kInvalidDeviceType) {
    joined_device_type = lhs->device_type();
    if (rhs->device_type() != kInvalidDeviceType && lhs->device_type() != rhs->device_type()) {
      return {};
    }
  } else {
    joined_device_type = rhs->device_type();
  }
  int joined_virtual_device_id;
  if (lhs->virtual_device_id >= 0) {
    joined_virtual_device_id = lhs->virtual_device_id;
    if (rhs->virtual_device_id >= 0 && lhs->virtual_device_id != rhs->virtual_device_id) {
      return {};
    }
  } else {
    joined_virtual_device_id = rhs->virtual_device_id;
  }
  Target joined_target;
  if (lhs->target.defined()) {
    joined_target = lhs->target;
    if (rhs->target.defined() && lhs->target != rhs->target) {
      return {};
    }
  } else {
    joined_target = rhs->target;
  }
  MemoryScope joined_memory_scope;
  if (!lhs->memory_scope.empty()) {
    joined_memory_scope = lhs->memory_scope;
    if (!rhs->memory_scope.empty() && lhs->memory_scope != rhs->memory_scope) {
      return {};
    }
  } else {
    joined_memory_scope = rhs->memory_scope;
  }
  return SEScope(joined_device_type, joined_virtual_device_id, joined_target, joined_memory_scope);
}

/* static */
SEScope SEScope::Default(const SEScope& lhs, const SEScope& rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  DLDeviceType defaulted_device_type;
  if (lhs->device_type() != kInvalidDeviceType) {
    defaulted_device_type = lhs->device_type();
  } else {
    defaulted_device_type = rhs->device_type();
  }
  int defaulted_virtual_device_id;
  if (lhs->virtual_device_id >= 0) {
    defaulted_virtual_device_id = lhs->virtual_device_id;
  } else {
    defaulted_virtual_device_id = rhs->virtual_device_id;
  }
  Target defaulted_target;
  if (lhs->target.defined()) {
    defaulted_target = lhs->target;
  } else {
    // We can only default to the rhs's target if it is consistent with the device type
    if (rhs->target.defined() && rhs->target->kind->device_type == defaulted_device_type) {
      defaulted_target = rhs->target;
    }
    // else: leave as null
  }
  MemoryScope defaulted_memory_scope;
  if (!lhs->memory_scope.empty()) {
    defaulted_memory_scope = lhs->memory_scope;
  } else {
    defaulted_memory_scope = rhs->memory_scope;
  }
  return SEScope(defaulted_device_type, defaulted_virtual_device_id, defaulted_target,
                 defaulted_memory_scope);
}

SEScope SEScopeCache::Make(DLDeviceType device_type, int virtual_device_id, Target target,
                           MemoryScope memory_scope) {
  SEScope prototype(device_type, virtual_device_id, std::move(target), std::move(memory_scope));
  auto itr = cache_.find(prototype);
  if (itr == cache_.end()) {
    VLOG(1) << "added new scope " << prototype;
    cache_.emplace(prototype);
    return prototype;
  } else {
    VLOG(1) << "reusing existing scope " << *itr;
    ICHECK_EQ(prototype->target.defined(), (*itr)->target.defined());
    if (prototype->target.defined()) {
      ICHECK_EQ(prototype->target->host.defined(), (*itr)->target->host.defined());
    }
    return *itr;
  }
}

SEScope SEScopeCache::Unique(const SEScope& scope) {
  return Make(scope->device_type(), scope->virtual_device_id, scope->target, scope->memory_scope);
}

TVM_REGISTER_GLOBAL("target.SEScope_ForDeviceTargetAndMemoryScope")
    .set_body_typed(SEScope::ForDeviceTargetAndMemoryScope);

}  // namespace tvm
