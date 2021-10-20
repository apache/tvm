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

void SEScopeNode::VisitAttrs(AttrVisitor* v) {
  int i = static_cast<int>(device_type_);
  v->Visit("device_type", &i);
  device_type_ = static_cast<DLDeviceType>(i);
  v->Visit("virtual_device_id", &virtual_device_id_);
  v->Visit("target", &target_);
  v->Visit("memory_scope", &memory_scope_);
}

bool SEScopeNode::SEqualReduce(const SEScopeNode* other, SEqualReducer equal) const {
  return device_type_ == other->device_type_ && virtual_device_id_ == other->virtual_device_id_ &&
         // NOTE: Comparing targets by their str representations
         target_->str() == other->target_->str() && memory_scope_ == other->memory_scope_;
}

void SEScopeNode::SHashReduce(SHashReducer hash_reduce) const {
  hash_reduce(device_type_);
  hash_reduce(virtual_device_id_);
  // NOTE: Reducing target to its str representation
  hash_reduce(target_->str());
  hash_reduce(memory_scope_);
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
  auto node = make_object<SEScopeNode>();
  node->device_type_ = device_type;
  node->virtual_device_id_ = virtual_device_id;
  node->target_ = std::move(target);
  node->memory_scope_ = std::move(memory_scope);
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
  return SEScope(joined_device_type, joined_virtual_device_id, joined_target, joined_memory_scope);
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
  return SEScope(defaulted_device_type, defaulted_virtual_device_id, defaulted_target,
                 defaulted_memory_scope);
}

SEScope SEScopeCache::Make(DLDeviceType device_type, int virtual_device_id, Target target,
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
  if (scope->is_fully_unconstrained()) {
    scope = SEScope::FullyUnconstrained();
  }
  cache_.emplace(key, scope);
  VLOG(1) << "new scope \"" << key << "\" -> " << scope;
  return scope;
}

SEScope SEScopeCache::Unique(const SEScope& scope) {
  return Make(scope->device_type(), scope->virtual_device_id(), scope->target(),
              scope->memory_scope());
}

TVM_REGISTER_GLOBAL("target.SEScope_ForDeviceTargetAndMemoryScope")
    .set_body_typed(SEScope::ForDeviceTargetAndMemoryScope);

}  // namespace tvm
