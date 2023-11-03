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
 * \file src/relay/analysis/device_domains.cc
 * \brief Unification domain for the device planner.
 */

#include "./device_domains.h"

#include <tvm/relay/attrs/call.h>
#include <tvm/relay/attrs/memory.h>

#include "../op/annotation/annotation.h"
#include "../op/call/call.h"
#include "../op/memory/device_copy.h"
#include "../op/memory/on_device.h"

namespace tvm {
namespace relay {
namespace transform {

DeviceDomains::DeviceDomains(CompilationConfig config) : config_(std::move(config)) {
  host_domain_ = MakeFirstOrderDomain(config_->host_virtual_device);
}

DeviceDomainPtr DeviceDomains::MakeFirstOrderDomain(const VirtualDevice& virtual_device) {
  if (virtual_device->IsFullyConstrained()) {
    auto itr = fully_constrained_virtual_device_to_domain_.find(virtual_device);
    if (itr != fully_constrained_virtual_device_to_domain_.end()) {
      return itr->second;
    }
    DeviceDomainPtr domain = std::make_shared<DeviceDomain>(virtual_device);
    fully_constrained_virtual_device_to_domain_.emplace(virtual_device, domain);
    return domain;
  } else {
    return std::make_shared<DeviceDomain>(virtual_device);
  }
}

DeviceDomainPtr DeviceDomains::MakeDomain(const Type& type, const VirtualDevice& virtual_device) {
  if (const auto* func_type_node = type.as<FuncTypeNode>()) {
    std::vector<DeviceDomainPtr> args_and_result;
    args_and_result.reserve(func_type_node->arg_types.size() + 1);
    for (const auto& arg_type : func_type_node->arg_types) {
      args_and_result.emplace_back(MakeDomain(arg_type, VirtualDevice::FullyUnconstrained()));
    }
    args_and_result.emplace_back(MakeDomain(func_type_node->ret_type, virtual_device));
    return std::make_shared<DeviceDomain>(std::move(args_and_result));
  } else {
    return MakeFirstOrderDomain(virtual_device);
  }
}

DeviceDomainPtr DeviceDomains::ForVirtualDevice(const Type& type,
                                                const VirtualDevice& non_canonical_virtual_device) {
  // Generally the virtual device will have come from an annotation so resolve it to ensure we have
  // its canonical representation.
  VirtualDevice virtual_device = config_->CanonicalVirtualDevice(non_canonical_virtual_device);
  ICHECK(!virtual_device->IsFullyUnconstrained());
  return MakeDomain(type, virtual_device);
}

DeviceDomainPtr DeviceDomains::Lookup(DeviceDomainPtr domain) {
  DeviceDomainPtr root = domain;
  while (true) {
    auto itr = domain_to_equiv_.find(root);
    if (itr == domain_to_equiv_.end()) {
      break;
    }
    ICHECK_NE(itr->second, root);
    root = itr->second;
    ICHECK_NOTNULL(root);
  }
  // Path compression.
  while (domain != root) {
    auto itr = domain_to_equiv_.find(domain);
    ICHECK(itr != domain_to_equiv_.end());
    domain = itr->second;
    ICHECK_NOTNULL(domain);
    itr->second = root;
  }
  return root;
}

DeviceDomainPtr DeviceDomains::JoinOrNull(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  ICHECK_EQ(lhs->args_and_result_.size(), rhs->args_and_result_.size())
      << "Device domains:" << std::endl
      << ToString(lhs) << std::endl
      << "and" << std::endl
      << ToString(rhs) << std::endl
      << "do not have the same kind and can't be unified.";
  if (lhs->args_and_result_.empty()) {
    // Directly compare first-order.
    if (rhs->virtual_device_->IsFullyUnconstrained()) {
      return lhs;
    }
    if (lhs->virtual_device_->IsFullyUnconstrained()) {
      return rhs;
    }
    Optional<VirtualDevice> joined_virtual_device =
        VirtualDevice::Join(lhs->virtual_device_, rhs->virtual_device_);
    if (!joined_virtual_device) {
      return nullptr;
    }
    return MakeFirstOrderDomain(config_->CanonicalVirtualDevice(joined_virtual_device.value()));
  } else {
    // Recurse for higher-order.
    std::vector<DeviceDomainPtr> args_and_result;
    args_and_result.reserve(lhs->args_and_result_.size());
    for (size_t i = 0; i < lhs->args_and_result_.size(); ++i) {
      DeviceDomainPtr joined_domain =
          UnifyOrNull(lhs->args_and_result_[i], rhs->args_and_result_[i]);
      if (joined_domain == nullptr) {
        return nullptr;
      }
      args_and_result.emplace_back(std::move(joined_domain));
    }
    return MakeHigherOrderDomain(std::move(args_and_result));
  }
}

DeviceDomainPtr DeviceDomains::UnifyOrNull(DeviceDomainPtr lhs, DeviceDomainPtr rhs) {
  ICHECK_NOTNULL(lhs);
  ICHECK_NOTNULL(rhs);
  lhs = Lookup(lhs);
  rhs = Lookup(rhs);
  DeviceDomainPtr joined_domain = JoinOrNull(lhs, rhs);
  if (joined_domain == nullptr) {
    return nullptr;
  }
  if (lhs != joined_domain) {
    domain_to_equiv_.emplace(lhs, joined_domain);
  }
  if (rhs != joined_domain) {
    domain_to_equiv_.emplace(rhs, joined_domain);
  }
  return joined_domain;
}

bool DeviceDomains::CollapseOrFalse(const DeviceDomainPtr& first_order_domain,
                                    const DeviceDomainPtr& higher_order_domain) {
  ICHECK(!first_order_domain->is_higher_order());
  ICHECK(higher_order_domain->is_higher_order());
  for (size_t i = 0; i < higher_order_domain->function_arity(); ++i) {
    if (UnifyOrNull(higher_order_domain->function_param(i), first_order_domain) == nullptr) {
      return false;
    }
  }
  return UnifyOrNull(higher_order_domain->function_result(), first_order_domain) != nullptr;
}

bool DeviceDomains::UnifyCollapsedOrFalse(const DeviceDomainPtr& lhs_first_order,
                                          const DeviceDomainPtr& rhs_maybe_higher_order) {
  ICHECK(!lhs_first_order->is_higher_order());
  if (rhs_maybe_higher_order->is_higher_order()) {
    return CollapseOrFalse(lhs_first_order, rhs_maybe_higher_order);
  } else {
    return UnifyOrNull(lhs_first_order, rhs_maybe_higher_order) != nullptr;
  }
}

DeviceDomainPtr DeviceDomains::DomainFor(const Expr& expr) {
  ICHECK(expr.defined());
  auto itr = expr_to_domain_.find(expr.get());
  if (itr != expr_to_domain_.end()) {
    return Lookup(itr->second);
  }
  auto domain = Free(expr->checked_type());
  expr_to_domain_.emplace(expr.get(), domain);
  return domain;
}

DeviceDomainPtr DeviceDomains::DomainForCallee(const Call& call) {
  auto itr = call_to_callee_domain_.find(call.get());
  if (itr != call_to_callee_domain_.end()) {
    return Lookup(itr->second);
  }
  std::vector<DeviceDomainPtr> args_and_result;

  OnDeviceProps on_device_props = GetOnDeviceProps(call.get());
  DeviceCopyProps device_copy_props = GetDeviceCopyProps(call.get());
  CallLoweredProps call_lowered_props = GetCallLoweredProps(call.get());

  if (call_lowered_props.lowered_func.defined()) {
    // Presumably we've already seen the call to the "primitive" Function from which this lowered
    // function was derived in an earlier PlanDevices pass. Thus we've already established that
    // all the argument and result devices domains must be equal, ignoring memory scopes.
    // So at this point we'll let all the arguments and result be free so that memory scopes can
    // differ.
    // TODO(mbs): As per header comments, need to revisit when can setup sub-virtual device
    // constraints.
    return DomainFor(call_lowered_props.lowered_func);
  } else if (on_device_props.body.defined()) {
    // By default:
    //   on_device(expr, virtual_device=<t>)
    //   on_device : fn(<t>):?x?
    // However we'll interpret the constrain_body and constrain_result fields to decide
    // on free vs constrained domains for the argument and result respectively.
    if (on_device_props.constrain_body) {
      args_and_result.emplace_back(
          ForVirtualDevice(on_device_props.body->checked_type(), on_device_props.virtual_device));
    } else {
      args_and_result.emplace_back(Free(on_device_props.body->checked_type()));
    }
    if (on_device_props.constrain_result) {
      args_and_result.emplace_back(
          ForVirtualDevice(on_device_props.body->checked_type(), on_device_props.virtual_device));
    } else {
      args_and_result.emplace_back(Free(on_device_props.body->checked_type()));
    }
  } else if (device_copy_props.body.defined()) {
    // device_copy(expr, src_virtual_device=<s>, dst_virtual_device=<d>)
    // device_copy: fn(<s>):<d>
    args_and_result.emplace_back(ForVirtualDevice(device_copy_props.body->checked_type(),
                                                  device_copy_props.src_virtual_device));
    args_and_result.emplace_back(ForVirtualDevice(device_copy_props.body->checked_type(),
                                                  device_copy_props.dst_virtual_device));
  } else if (call->op == alloc_storage_op) {
    ICHECK_EQ(call->args.size(), 3U);
    // alloc_storage(size, shape, alignment, virtual_device=<t>)
    // alloc_storage: fn(<cpu>, <cpu>, <cpu>):<t>
    const auto* attrs = call->attrs.as<AllocStorageAttrs>();
    args_and_result.emplace_back(host_domain_);
    args_and_result.emplace_back(host_domain_);
    args_and_result.emplace_back(host_domain_);
    args_and_result.emplace_back(ForVirtualDevice(call->checked_type(), attrs->virtual_device));
  } else if (call->op == alloc_tensor_op) {
    ICHECK_EQ(call->args.size(), 3U);
    // alloc_tensor(storage, offset, shape)
    // alloc_tensor: fn(?x?, <cpu>, <cpu>):?x?
    auto free_domain = Free(call->checked_type());
    args_and_result.emplace_back(free_domain);
    args_and_result.emplace_back(host_domain_);
    args_and_result.emplace_back(host_domain_);
    args_and_result.emplace_back(free_domain);
  } else if (call->op == shape_of_op) {
    ICHECK_EQ(call->args.size(), 1U);
    // shape_of(tensor)
    // shape_of: fn(?x?):<cpu>
    args_and_result.emplace_back(Free(call->args[0]->checked_type()));
    args_and_result.emplace_back(host_domain_);
  } else if (call->op == invoke_tvm_op) {
    ICHECK_EQ(call->args.size(), 3U);
    // invoke_tvm_op(op, inputs, outputs)
    // invoke_tvm_op: fn(..., ?x?, ?x?):?x?
    // where ... is a free domain appropriate for op's type
    auto free_domain = Free(call->checked_type());
    args_and_result.emplace_back(Free(call->args[0]->checked_type()));
    args_and_result.emplace_back(free_domain);
    args_and_result.emplace_back(free_domain);
    args_and_result.emplace_back(free_domain);
  } else if (call->op == reshape_tensor_op) {
    ICHECK_EQ(call->args.size(), 2U);
    // reshape_tensor(data, shape)
    // reshape_tensor: fn(?x?, <cpu>):?x?
    auto free_domain = Free(call->checked_type());
    args_and_result.emplace_back(free_domain);
    args_and_result.emplace_back(host_domain_);
    args_and_result.emplace_back(free_domain);
  } else if (call->op->IsInstance<OpNode>()) {
    // <primitive>(arg1, ..., argn)
    // <primitive>: fn(?x?, ..., ?x?):?x?
    // (all args and result must be first-order).
    auto free_domain = MakeFirstOrderDomain(VirtualDevice::FullyUnconstrained());
    for (size_t i = 0; i < call->args.size(); ++i) {
      args_and_result.emplace_back(free_domain);
    }
    args_and_result.emplace_back(free_domain);
  } else if (call->op->IsInstance<ConstructorNode>()) {
    // <constructor>(arg1, ..., argn)
    // <constructor>: fn(?x1?, ..., ?xn?):?xr?
    // where we force all possibly higher-order ?xi? to be collapsed to the first-order ?xr?.
    // TODO(mbs): This assumes we've eta-expanded constructors, thus all constructors appear
    // in callee positions.
    const auto* func_type_node = call->op->checked_type().as<FuncTypeNode>();
    ICHECK_NOTNULL(func_type_node);
    ICHECK_EQ(func_type_node->arg_types.size(), call->args.size());
    auto result_domain = Free(func_type_node->ret_type);  // first-order
    for (const auto& arg_type : func_type_node->arg_types) {
      auto param_domain = Free(arg_type);                                 // possibly higher-order
      bool success = UnifyCollapsedOrFalse(result_domain, param_domain);  // collapse if required
      ICHECK(success);
      args_and_result.emplace_back(param_domain);
    }
    args_and_result.emplace_back(result_domain);
  } else {
    // We still need to handle the case where the function / op is not lowered
    // because the device planner runs both before and after lowering.
    return DomainFor(call->op);
  }
  auto domain = MakeHigherOrderDomain(std::move(args_and_result));
  call_to_callee_domain_.emplace(call.get(), domain);
  return domain;
}

void DeviceDomains::UnifyExprExact(const Expr& lhs, const Expr& rhs) {
  auto lhs_domain = DomainFor(lhs);
  auto rhs_domain = DomainFor(rhs);
  if (UnifyOrNull(lhs_domain, rhs_domain) == nullptr) {
    // TODO(mbs): Proper diagnostics.
    LOG(FATAL) << "Incompatible virtual devices for expressions:" << std::endl
               << PrettyPrint(lhs) << std::endl
               << "with virtual device:" << std::endl
               << ToString(lhs_domain) << "and:" << std::endl
               << PrettyPrint(rhs) << std::endl
               << "with virtual device:" << std::endl
               << ToString(rhs_domain);
  }
}

void DeviceDomains::OptionalUnifyExprExact(const Expr& lhs, const Expr& rhs) {
  auto lhs_domain = DomainFor(lhs);
  auto rhs_domain = DomainFor(rhs);
  // Snapshot
  std::unordered_map<DeviceDomainPtr, DeviceDomainPtr> domain_to_equiv_snapshot = domain_to_equiv_;
  if (UnifyOrNull(lhs_domain, rhs_domain) == nullptr) {
    // Rollback
    domain_to_equiv_ = domain_to_equiv_snapshot;
    VLOG(2) << "Unable to unify virtual devices for expression:" << std::endl
            << PrettyPrint(lhs) << std::endl
            << "with virtual device:" << std::endl
            << ToString(lhs_domain) << std::endl
            << "and expression:" << std::endl
            << PrettyPrint(rhs) << std::endl
            << "with virtual device:" << std::endl
            << ToString(rhs_domain) << std::endl
            << ". Leaving virtual devices non-unified.";
  } else {
    VLOG(2) << "Unified virtual devices for expression:" << std::endl
            << PrettyPrint(lhs) << std::endl
            << "and expression:" << std::endl
            << PrettyPrint(rhs) << std::endl
            << "to virtual devices:" << std::endl
            << ToString(lhs_domain);
  }
}

void DeviceDomains::UnifyExprExact(const Expr& expr, const DeviceDomainPtr& expected_domain) {
  auto actual_domain = DomainFor(expr);
  if (UnifyOrNull(actual_domain, expected_domain) == nullptr) {
    // TODO(mbs): Proper diagnostics.
    LOG(FATAL) << "Incompatible virtual devices for expression:" << std::endl
               << PrettyPrint(expr) << std::endl
               << "with actual virtual device:" << std::endl
               << ToString(actual_domain) << std::endl
               << "and expected virtual device:" << std::endl
               << ToString(expected_domain);
  }
}

void DeviceDomains::UnifyExprCollapsed(const Expr& expr_first_order,
                                       const DeviceDomainPtr& expected_domain_maybe_higher_order) {
  auto actual_domain_first_order = DomainFor(expr_first_order);
  if (!UnifyCollapsedOrFalse(actual_domain_first_order, expected_domain_maybe_higher_order)) {
    // TODO(mbs): Proper diagnostics.
    LOG(FATAL) << "Incompatible virtual devices for expression:" << std::endl
               << PrettyPrint(expr_first_order) << std::endl
               << "with actual virtual devices:" << std::endl
               << ToString(actual_domain_first_order) << std::endl
               << "and expected virtual device:" << std::endl
               << ToString(expected_domain_maybe_higher_order);
  }
}

bool DeviceDomains::IsFullyConstrained(DeviceDomainPtr domain) {
  domain = Lookup(domain);
  if (domain->args_and_result_.empty()) {
    // First-order.
    return domain->virtual_device_->IsFullyConstrained();
  } else {
    // Higher-order.
    return std::all_of(
        domain->args_and_result_.begin(), domain->args_and_result_.end(),
        [this](const DeviceDomainPtr& sub_domain) { return IsFullyConstrained(sub_domain); });
  }
}

void DeviceDomains::SetDefault(DeviceDomainPtr domain,
                               const VirtualDevice& default_virtual_device) {
  ICHECK(!default_virtual_device->IsFullyUnconstrained());
  domain = Lookup(domain);
  if (domain->args_and_result_.empty()) {
    DeviceDomainPtr default_domain = MakeFirstOrderDomain(config_->CanonicalVirtualDevice(
        VirtualDevice::Default(domain->virtual_device_, default_virtual_device)));
    DeviceDomainPtr defaulted_domain_ptr = UnifyOrNull(domain, default_domain);
    ICHECK(defaulted_domain_ptr != nullptr) << "domain:" << std::endl
                                            << ToString(domain) << std::endl
                                            << "default domain:" << std::endl
                                            << ToString(default_domain);
  } else {
    for (const auto& sub_domain : domain->args_and_result_) {
      SetDefault(sub_domain, default_virtual_device);
    }
  }
}

void DeviceDomains::SetResultDefaultThenParams(const DeviceDomainPtr& domain_maybe_higher_order,
                                               const VirtualDevice& default_virtual_device) {
  if (domain_maybe_higher_order->args_and_result_.empty()) {
    SetDefault(domain_maybe_higher_order, default_virtual_device);
  } else {
    // First set default for result domain.
    SetDefault(ResultDomain(domain_maybe_higher_order), default_virtual_device);
    // Then use current result domain as default for everything else.
    SetDefault(domain_maybe_higher_order, ResultVirtualDevice(domain_maybe_higher_order));
  }
}

DeviceDomainPtr DeviceDomains::ResultDomain(DeviceDomainPtr domain) {
  domain = Lookup(domain);
  while (!domain->args_and_result_.empty()) {
    domain = Lookup(domain->args_and_result_.back());
  }
  return domain;
}

std::string DeviceDomains::ToString(DeviceDomainPtr domain) {
  domain = Lookup(domain);
  std::ostringstream os;
  if (domain->args_and_result_.empty()) {
    // First-order.
    if (!domain->virtual_device_->IsFullyConstrained()) {
      os << "?" << static_cast<size_t>(reinterpret_cast<uintptr_t>(domain.get())) << "?";
    }
    if (!domain->virtual_device_->IsFullyUnconstrained()) {
      os << domain->virtual_device_;
    }
  } else {
    // higher-order
    os << "fn(";
    for (size_t i = 0; i + 1 < domain->args_and_result_.size(); ++i) {
      if (i > 0) {
        os << ",";
      }
      os << ToString(domain->args_and_result_[i]);
    }
    os << "):" << ToString(domain->args_and_result_.back());
  }
  return os.str();
}

std::string DeviceDomains::ToString() {
  std::ostringstream os;
  for (const auto& pair : expr_to_domain_) {
    os << "expression:" << std::endl
       << PrettyPrint(GetRef<Expr>(pair.first)) << std::endl
       << "domain:" << std::endl
       << ToString(pair.second) << std::endl
       << std::endl;
  }
  for (const auto& pair : call_to_callee_domain_) {
    os << "call:" << std::endl
       << PrettyPrint(GetRef<Call>(pair.first)) << std::endl
       << "callee domain:" << std::endl
       << ToString(pair.second) << std::endl
       << std::endl;
  }
  return os.str();
}

}  // namespace transform
}  // namespace relay
}  // namespace tvm
