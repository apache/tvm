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

namespace tvm {
namespace relay {
namespace transform {

namespace {

// Ye olde boost hash mixer.
constexpr size_t mix(size_t h1, size_t h2) {
  return h1 ^ (h1 + 0x9e3779b9 + (h2 << 6) + (h2 >> 2));
}

/*!
 * \brief As for GetDeviceCopyProps, but for the call to the lowered TIR primitives rather
 * than the original "device_copy" operator.
 *
 * See te_compiler.cc for where this rewriting occurs.
 */
DeviceCopyProps GetPrimitiveDeviceCopyProps(const CallNode* call_node) {
  if (call_node->op == CallLoweredOp()) {
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);
    if (call_lowered_props.attrs.metadata.count("source_device") == 1 &&
        call_lowered_props.attrs.metadata.count("dst_device") == 1) {
      ICHECK_EQ(call_lowered_props.arguments.size(), 1) << "device_copy is of arity 1";
      return {call_lowered_props.arguments[0],
              static_cast<DLDeviceType>(
                  Downcast<Integer>(call_lowered_props.attrs.metadata["source_device"])->value),
              static_cast<DLDeviceType>(
                  Downcast<Integer>(call_lowered_props.attrs.metadata["dst_device"])->value)};
    }
  }
  return {};
}

}  // namespace

// The following hash and equality helpers give each free first-order domain pointer its own
// distinct identity.

size_t DeviceDomainHash::operator()(const DeviceDomainPtr& domain) const {
  if (domain->is_free()) {
    // Give each free first-order domain its own identity.
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(domain.get()));
  } else {
    size_t h = domain->args_and_result_.size();
    h = mix(h, std::hash<int>()(static_cast<int>(domain->device_type_)));
    for (const auto& sub_domain_ptr : domain->args_and_result_) {
      h = mix(h, DeviceDomainHash()(sub_domain_ptr));
    }
    return h;
  }
}

bool DeviceDomainEqual::operator()(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) const {
  if (lhs->args_and_result_.size() != rhs->args_and_result_.size()) {
    // Mismatched arities are never equal.
    // (Though we'll never ask to do such a comparison explicitly, the hash map
    // may do so implicitly due to hash collisions.)
    return false;
  }
  if (lhs->is_free() && rhs->is_free()) {
    // Compare first-order free domains by their address.
    return lhs.get() == rhs.get();
  }
  if (lhs->args_and_result_.empty()) {
    // Compare first-order domains by their device type -- free vs bound will compare as false.
    return lhs->device_type_ == rhs->device_type_;
  } else {
    // Compare higher-order domains pointwise.
    for (size_t i = 0; i < lhs->args_and_result_.size(); ++i) {
      if (!(*this)(lhs->args_and_result_[i], rhs->args_and_result_[i])) {
        return false;
      }
    }
    return true;
  }
}

/* static */
DeviceDomainPtr DeviceDomains::MakeDomain(const Type& type, DLDeviceType device_type) {
  if (const auto* func_type_node = type.as<FuncTypeNode>()) {
    std::vector<DeviceDomainPtr> args_and_result;
    args_and_result.reserve(func_type_node->arg_types.size() + 1);
    for (const auto& arg_type : func_type_node->arg_types) {
      args_and_result.emplace_back(MakeDomain(arg_type, kInvalidDeviceType));
    }
    args_and_result.emplace_back(MakeDomain(func_type_node->ret_type, device_type));
    return std::make_shared<DeviceDomain>(std::move(args_and_result));
  } else {
    return std::make_shared<DeviceDomain>(device_type);
  }
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

DeviceDomainPtr DeviceDomains::Join(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) {
  // TODO(mbs): Proper diagnostics.
  ICHECK_EQ(lhs->args_and_result_.size(), rhs->args_and_result_.size())
      << "Device domains:" << std::endl
      << ToString(lhs) << std::endl
      << "and" << std::endl
      << ToString(rhs) << std::endl
      << "do not have the same kind and can't be unified.";
  if (rhs->is_free()) {
    return lhs;
  } else if (lhs->is_free()) {
    return rhs;
  } else if (lhs->args_and_result_.empty()) {
    // Must have consistent device types for first order domains.
    if (lhs->device_type_ != rhs->device_type_) {
      // TODO(mbs): Proper diagnostics.
      std::ostringstream os;
      os << "Inconsistent device types " << lhs->device_type_ << " and " << rhs->device_type_;
      throw Error(os.str());
    }
    return lhs;
  } else {
    // Recurse for higher-order.
    std::vector<DeviceDomainPtr> args_and_result;
    args_and_result.reserve(lhs->args_and_result_.size());
    for (size_t i = 0; i < lhs->args_and_result_.size(); ++i) {
      args_and_result.emplace_back(Unify(lhs->args_and_result_[i], rhs->args_and_result_[i]));
    }
    return MakeDomain(std::move(args_and_result));
  }
}

DeviceDomainPtr DeviceDomains::Unify(DeviceDomainPtr lhs, DeviceDomainPtr rhs) {
  lhs = Lookup(lhs);
  rhs = Lookup(rhs);
  auto joined_domain = Join(lhs, rhs);
  if (!DeviceDomainEqual()(lhs, joined_domain)) {
    domain_to_equiv_.emplace(lhs, joined_domain);
  }
  if (!DeviceDomainEqual()(rhs, joined_domain)) {
    domain_to_equiv_.emplace(rhs, joined_domain);
  }
  return joined_domain;
}

void DeviceDomains::UnifyCollapsed(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) {
  if (!lhs->is_higher_order() && rhs->is_higher_order()) {
    Collapse(lhs, rhs);
  } else {
    Unify(lhs, rhs);
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

  auto on_device_props = GetOnDeviceProps(call.get());
  auto device_copy_props = GetDeviceCopyProps(call.get());
  if (!device_copy_props.body.defined()) {
    device_copy_props = GetPrimitiveDeviceCopyProps(call.get());
  }

  if (on_device_props.body.defined()) {
    // on_device(expr, device_type=<t>, is_fixed=false)
    // on_device : fn(<t>):?x?
    //
    // on_device(expr, device_type=<t>, is_fixed=true)
    // on_device: fn(<t>):<t>
    args_and_result.emplace_back(
        ForDeviceType(on_device_props.body->checked_type(), on_device_props.device_type));
    if (on_device_props.is_fixed) {
      args_and_result.emplace_back(args_and_result.front());
    } else {
      args_and_result.emplace_back(Free(on_device_props.body->checked_type()));
    }
  } else if (device_copy_props.body.defined()) {
    // device_copy(expr, src_dev_type=<s>, dst_dev_type=<d>)
    // device_copy: fn(<s>):<d>
    args_and_result.emplace_back(
        ForDeviceType(device_copy_props.body->checked_type(), device_copy_props.src_dev_type));
    args_and_result.emplace_back(
        ForDeviceType(device_copy_props.body->checked_type(), device_copy_props.dst_dev_type));
  } else if (call->op == alloc_storage_op) {
    ICHECK_EQ(call->args.size(), 2U);
    // alloc_storage(size, alignment, device_type=<t>)
    // alloc_storage: fn(<cpu>, <cpu>):<t>
    const auto* attrs = call->attrs.as<AllocStorageAttrs>();
    args_and_result.emplace_back(cpu_domain_);
    args_and_result.emplace_back(cpu_domain_);
    args_and_result.emplace_back(
        ForDeviceType(call->checked_type(), static_cast<DLDeviceType>(attrs->device_type)));
  } else if (call->op == alloc_tensor_op) {
    ICHECK_EQ(call->args.size(), 3U);
    // alloc_tensor(storage, offset, shape)
    // alloc_tensor: fn(?x?, <cpu>, <cpu>):?x?
    auto free_domain = Free(call->checked_type());
    args_and_result.emplace_back(free_domain);
    args_and_result.emplace_back(cpu_domain_);
    args_and_result.emplace_back(cpu_domain_);
    args_and_result.emplace_back(free_domain);
  } else if (call->op == shape_func_op) {
    ICHECK_EQ(call->args.size(), 3U);
    // shape_func(func, inputs, outputs, is_inputs=[...])
    // shape_func: fn(..., <cpu>, <cpu>):<cpu>
    // where ... is a free domain appropriate for func's type
    args_and_result.emplace_back(Free(call->args[0]->checked_type()));
    // TODO(mbs): I think this should be on the cpu only when is_input = [false], but
    // what do we do when we have multiple arguments with different is_input values?
    args_and_result.emplace_back(cpu_domain_);
    args_and_result.emplace_back(cpu_domain_);
    args_and_result.emplace_back(cpu_domain_);
  } else if (call->op == shape_of_op) {
    ICHECK_EQ(call->args.size(), 1U);
    // shape_of(tensor)
    // shape_of: fn(?x?):<cpu>
    args_and_result.emplace_back(Free(call->args[0]->checked_type()));
    args_and_result.emplace_back(cpu_domain_);
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
    args_and_result.emplace_back(cpu_domain_);
    args_and_result.emplace_back(free_domain);
  } else if (call->op->IsInstance<OpNode>()) {
    // <primitive>(arg1, ..., argn)
    // <primitive>: fn(?x?, ..., ?x?):?x?
    // (all args and result must be first-order).
    auto free_domain = Free(arb_);
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
      auto param_domain = Free(arg_type);           // possibly higher-order
      UnifyCollapsed(result_domain, param_domain);  // collapse if required
      args_and_result.emplace_back(param_domain);
    }
    args_and_result.emplace_back(result_domain);
  } else if (call->op == CallLoweredOp()) {
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call.get());
    return DomainFor(call_lowered_props.lowered_func);
  } else {
    // We still need to handle the case where the function / op is not lowered
    // because the device planner runs before and after lowering.
    return DomainFor(call->op);
  }
  auto domain = MakeDomain(std::move(args_and_result));
  call_to_callee_domain_.emplace(call.get(), domain);
  return domain;
}

void DeviceDomains::UnifyExprExact(const Expr& lhs, const Expr& rhs) {
  auto lhs_domain = DomainFor(lhs);
  auto rhs_domain = DomainFor(rhs);
  try {
    Unify(lhs_domain, rhs_domain);
  } catch (const Error& e) {
    // TODO(mbs): Proper diagnostics.
    LOG(FATAL) << "Incompatible devices for expressions:" << std::endl
               << PrettyPrint(lhs) << std::endl
               << "with device:" << std::endl
               << ToString(lhs_domain) << "and:" << std::endl
               << PrettyPrint(rhs) << std::endl
               << "with device:" << std::endl
               << ToString(rhs_domain) << std::endl
               << e.what();
  }
}

void DeviceDomains::UnifyExprExact(const Expr& expr, const DeviceDomainPtr& expected_domain) {
  auto actual_domain = DomainFor(expr);
  try {
    Unify(actual_domain, expected_domain);
  } catch (const Error& e) {
    // TODO(mbs): Proper diagnostics.
    LOG(FATAL) << "Incompatible devices for expression:" << std::endl
               << PrettyPrint(expr) << std::endl
               << "with actual device:" << std::endl
               << ToString(actual_domain) << std::endl
               << "and expected device:" << std::endl
               << ToString(expected_domain) << std::endl
               << e.what();
  }
}

void DeviceDomains::UnifyExprCollapsed(const Expr& expr, const DeviceDomainPtr& expected_domain) {
  auto actual_domain = DomainFor(expr);
  try {
    UnifyCollapsed(actual_domain, expected_domain);
  } catch (const Error& e) {
    // TODO(mbs): Proper diagnostics.
    LOG(FATAL) << "Incompatible devices for expression:" << std::endl
               << PrettyPrint(expr) << std::endl
               << "with actual device:" << std::endl
               << ToString(actual_domain) << std::endl
               << "and expected device:" << std::endl
               << ToString(expected_domain) << std::endl
               << e.what();
  }
}

bool DeviceDomains::AnyFree(DeviceDomainPtr domain) {
  domain = Lookup(domain);
  if (domain->is_free()) {
    return true;
  }
  for (const auto& sub_domain : domain->args_and_result_) {
    if (AnyFree(sub_domain)) {
      return true;
    }
  }
  return false;
}

void DeviceDomains::Collapse(const DeviceDomainPtr& first_order_domain,
                             const DeviceDomainPtr& higher_order_domain) {
  for (size_t i = 0; i < higher_order_domain->function_arity(); ++i) {
    Unify(higher_order_domain->function_param(i), first_order_domain);
  }
  Unify(higher_order_domain->function_result(), first_order_domain);
}

void DeviceDomains::SetDefault(DeviceDomainPtr domain, DLDeviceType default_device_type) {
  ICHECK_NE(default_device_type, kInvalidDeviceType);
  domain = Lookup(domain);
  if (domain->is_free()) {
    // Will never throw since lhs is free.
    Unify(domain, std::make_shared<DeviceDomain>(default_device_type));
  } else if (!domain->args_and_result_.empty()) {
    for (const auto& sub_domain : domain->args_and_result_) {
      SetDefault(sub_domain, default_device_type);
    }
  }
}

void DeviceDomains::SetResultDefaultThenParams(const DeviceDomainPtr& domain,
                                               DLDeviceType default_device_type) {
  if (!domain->is_higher_order()) {
    SetDefault(domain, default_device_type);
    return;
  }
  DLDeviceType result_device_type = ResultDeviceType(domain);
  if (result_device_type == kInvalidDeviceType) {
    // If the function result device is still free use the given default.
    result_device_type = default_device_type;
  }
  // Default any remaining free parameters to the function result device.
  SetDefault(domain, result_device_type);
}

std::string DeviceDomains::ToString(DeviceDomainPtr domain) {
  domain = Lookup(domain);
  std::ostringstream os;
  if (domain->is_free()) {
    // first-order free
    os << "?" << static_cast<size_t>(reinterpret_cast<uintptr_t>(domain.get())) << "?";
  } else if (domain->args_and_result_.empty()) {
    // first-order bound
    os << "<" << domain->device_type_ << ">";
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

DeviceDomainPtr DeviceDomains::ResultDomain(DeviceDomainPtr domain) {
  domain = Lookup(domain);
  while (!domain->args_and_result_.empty()) {
    domain = Lookup(domain->args_and_result_.back());
  }
  return domain;
}

}  // namespace transform
}  // namespace relay
}  // namespace tvm
