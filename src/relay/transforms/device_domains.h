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
 * \file src/relay/analysis/device_domains.h
 * \brief Unification domain for the device planner.
 */

#ifndef TVM_RELAY_TRANSFORMS_DEVICE_DOMAINS_H_
#define TVM_RELAY_TRANSFORMS_DEVICE_DOMAINS_H_

#include <dlpack/dlpack.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/ndarray.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace transform {

class DeviceDomain;
using DeviceDomainPtr = std::shared_ptr<DeviceDomain>;

/*!
 * \brief Represents the domain over which we collect equality constraints.
 *
 * \code
 *   D ::= ?x?                  -- first order, free
 *       | <device_type>        -- first order, bound
 *       | fn(D1, ..., Dn):Dr   -- higher order
 * \endcode
 *
 * We require a function value to be on the same device as its result. To support that we need
 * a notion of the 'result domain' of a domain:
 * \code
 *   result_domain(?x?)                = ?x?
 *   result_domain(<device_type>)      = <device_type>
 *   result_domain(fn(D1, ..., Dn):Dr) = result_domain(Dr)
 * \endcode
 */
class DeviceDomain {
 public:
  /*!
   * \brief Constructs a first-order domain of \p device_type, which may be
   * \p kInvalidDeviceType to indicate the domain is free.
   */
  explicit DeviceDomain(DLDeviceType device_type) : device_type_(device_type) {}

  /*!
   * \brief Constructs a higher-order domain, where \p args_and_result contain the
   * function argument and result domains in order.
   */
  explicit DeviceDomain(std::vector<DeviceDomainPtr> args_and_result)
      : device_type_(kInvalidDeviceType), args_and_result_(std::move(args_and_result)) {}

  /*! \brief Returns true if domain is first-order and free. */
  bool is_free() const { return device_type_ == kInvalidDeviceType && args_and_result_.empty(); }

  /*! \brief Returns true if domain is higher-order. */
  bool is_higher_order() const { return !args_and_result_.empty(); }

  DLDeviceType first_order_device_type() const {
    ICHECK(args_and_result_.empty());
    return device_type_;
  }

  size_t function_arity() const {
    ICHECK(!args_and_result_.empty());
    return args_and_result_.size() - 1UL;
  }

  DeviceDomainPtr function_param(size_t i) const {
    ICHECK(!args_and_result_.empty());
    ICHECK_LT(i + 1, args_and_result_.size());
    return args_and_result_[i];
  }

  DeviceDomainPtr function_result() const {
    ICHECK(!args_and_result_.empty());
    return args_and_result_.back();
  }

 private:
  /*!
   * \brief If this is a function domain then always kInvalidDevice. Otherwise will be
   * kInvalidDevice if the domain is still free, or the specific concrete device if the domain is
   * bound.
   */
  const DLDeviceType device_type_;

  /*!
   * \brief If this is a function domain then the sub-domains for each of the function's
   * arguments, and the domain for its result. Otherwise empty.
   */
  const std::vector<DeviceDomainPtr> args_and_result_;

  friend struct DeviceDomainHash;
  friend struct DeviceDomainEqual;
  friend class DeviceDomains;
};

// The following hash and equality helpers give each free first-order domain pointer its own
// distinct identity.
struct DeviceDomainHash {
  size_t operator()(const DeviceDomainPtr& domain) const;
};

struct DeviceDomainEqual {
 public:
  bool operator()(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs) const;
};

/*!
 * \brief Tracks the device domains for a set of expressions w.r.t. an equivalence relation
 * built up by calls to \p Unify.
 */
class DeviceDomains {
 public:
  DeviceDomains() = default;

  /*!
   * \brief Returns a domain appropriate for \p type who's result domain is bound
   * to \p device_type. If \p device_type is \p kInvalidDeviceType then the entire domain
   * will be free.
   */
  static DeviceDomainPtr MakeDomain(const Type& type, DLDeviceType device_type);

  /*!
   * \brief Returns a higher-order domain with \p args_and_results.
   */
  static DeviceDomainPtr MakeDomain(std::vector<DeviceDomainPtr> arg_and_results) {
    return std::make_shared<DeviceDomain>(std::move(arg_and_results));
  }

  /*! \brief Returns a domain with the given result device type appropriate \p device_type. */
  static DeviceDomainPtr ForDeviceType(const Type& type, DLDeviceType device_type) {
    ICHECK_NE(device_type, kInvalidDeviceType);
    return MakeDomain(type, device_type);
  }

  /*! \brief Returns a free domain appropriate for \p type. */
  static DeviceDomainPtr Free(const Type& type) { return MakeDomain(type, kInvalidDeviceType); }

  /*! \brief Returns the domain representing the equivalence class containing \p domain. */
  DeviceDomainPtr Lookup(DeviceDomainPtr domain);

  /*!
   * \brief Returns the domain accounting for all bound devices in \p lhs and \p rhs.
   *
   * Throws \p Error on failure.
   */
  DeviceDomainPtr Join(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs);

  /*!
   * \brief Unifies \p lhs and \p rhs, returning the most-bound of the two. Fails if \p lhs and \p
   * rhs disagree on bound device type.
   *
   * Throws \p Error on failure.
   */
  // TODO(mbs): I don't think we need an occurs check since the program is well-typed, but
  // given we have refs to functions I'm prepared to be surprised.
  DeviceDomainPtr Unify(DeviceDomainPtr lhs, DeviceDomainPtr rhs);

  /*!
   * \brief Unifies \p lhs and \p rhs. If \p lhs is first-order and \p rhs is higher-order,
   * require all arguments and result of \p rhs to unify with \p lhs. Otherwise same as
   * \p Unify.
   *
   * Throws \p Error on failure.
   */
  void UnifyCollapsed(const DeviceDomainPtr& lhs, const DeviceDomainPtr& rhs);

  /*! \brief Returns true if a domain is known for \p expr. */
  bool contains(const Expr& expr) const { return expr_to_domain_.count(expr.get()); }

  /*! \brief Returns the domain representing \p expr. */
  DeviceDomainPtr DomainFor(const Expr& expr);

  /*!
   * \brief Returns the domain representing the callee (ie 'op') in \p call expression. If the
   * callee is a primitive or special operation we handle it specially. Otherwise defers to \p
   * DomainFor(call->op).
   *
   * This special handling is needed:
   * - To handle the "on_device" and "device_copy" ops which constrain devices to the given devices.
   * - To handle some special ops which constrain devices to the CPU.
   * - To allow the same primitive to be called on different devices at different call sites.
   * Since each call to the op can have a different domain we index the ops by the call expression
   * rather than the op itself.
   */
  DeviceDomainPtr DomainForCallee(const Call& call);

  /*! \brief Unifies the domains for expressions \p lhs and \p rhs. */
  void UnifyExprExact(const Expr& lhs, const Expr& rhs);

  /*!
   * \brief Unifies the domain for \p expr with \p expected_domain.
   */
  void UnifyExprExact(const Expr& expr, const DeviceDomainPtr& expected_domain);

  /*!
   * \brief Unifies the domain for \p expr with \p expected_domain.
   * If \p expected_domain is higher-order but \p expr is first-order, require all arguments
   * and the result of \p expected_domain to have the same domain as for \p expr.
   */
  void UnifyExprCollapsed(const Expr& expr, const DeviceDomainPtr& expected_domain);

  /*! \brief Returns true if \p domain contains any free sub-domains. */
  bool AnyFree(DeviceDomainPtr domain);

  /*
   * \brief Force all domains in \p higher_order_domain to unify with \p first_order_domain.
   * This can be used to handle functions within tuples, references and ADTs since we don't
   * attempt to track anything beyond 'the device' for expressions of those first-order types.
   *
   * Throws \p Error on failure.
   */
  void Collapse(const DeviceDomainPtr& first_order_domain,
                const DeviceDomainPtr& higher_order_domain);

  /*! \brief Force all free domains in \p domain to default to \p default_device_type. */
  void SetDefault(DeviceDomainPtr domain, DLDeviceType default_device_type);

  /*!
   * \brief If \p domain is higher-order and its result domain is free, force it to
   * \p default_device_type. Then force any  remaining free domains to the result domain
   * (freshly defaulted or original). If \p domain is first-order same as \p SetDefault.
   */
  void SetResultDefaultThenParams(const DeviceDomainPtr& domain, DLDeviceType default_device_type);

  /*! \brief Returns one-line description of \p domain for debugging. */
  std::string ToString(DeviceDomainPtr domain);

  /*! \brief Returns description of entire system of constraints for debugging */
  std::string ToString();

  /*!
   * \brief Returns the result domain for \p domain (see defn in DeviceDomain comment).
   */
  DeviceDomainPtr ResultDomain(DeviceDomainPtr domain);

  /*!
   * \brief Returns the result (possibly free) device type for \p domain (see defn in DeviceDomain
   * comment).
   */
  DLDeviceType ResultDeviceType(const DeviceDomainPtr& domain) {
    return ResultDomain(domain)->first_order_device_type();
  }

 private:
  /*! \brief Intrinsics we need to handle specially. */
  const Op& alloc_storage_op = Op::Get("memory.alloc_storage");
  const Op& alloc_tensor_op = Op::Get("memory.alloc_tensor");
  const Op& shape_of_op = Op::Get("vm.shape_of");
  const Op& invoke_tvm_op = Op::Get("vm.invoke_tvm_op");
  const Op& shape_func_op = Op::Get("vm.shape_func");
  const Op& reshape_tensor_op = Op::Get("vm.reshape_tensor");
  /*! \brief The CPU device type for special operators such as dynamic shape functions. */
  const DLDeviceType cpu_device_type_ = kDLCPU;
  /*! \brief Placeholder for any first-order type. */
  Type arb_ = TupleType();
  /*! \brief The domain for first-order expressions on the CPU. */
  DeviceDomainPtr cpu_domain_ = ForDeviceType(arb_, cpu_device_type_);

  /*! \brief Maps expressions to their domains as determined during analysis. */
  std::unordered_map<const ExprNode*, DeviceDomainPtr> expr_to_domain_;

  /*!
   * \brief Maps call expressions to the domains for their callee where the callee is a primitive.
   */
  std::unordered_map<const CallNode*, DeviceDomainPtr> call_to_callee_domain_;

  /*! \brief Maps device domains to their equivalent domains as determined during unification. */
  std::unordered_map<DeviceDomainPtr, DeviceDomainPtr, DeviceDomainHash, DeviceDomainEqual>
      domain_to_equiv_;
};

}  // namespace transform
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_DEVICE_DOMAINS_H_
