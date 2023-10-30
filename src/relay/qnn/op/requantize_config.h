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
 * \file src/relay/qnn/op/requantize_config.h
 * \brief QNN requantize config.
 */

#ifndef TVM_RELAY_QNN_OP_REQUANTIZE_CONFIG_H_
#define TVM_RELAY_QNN_OP_REQUANTIZE_CONFIG_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include <string>

#include "../../op/op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

class RequantizeConfig;
/*!
 * \brief Container for build configuration options
 */
class RequantizeConfigNode : public Object {
  std::string rounding;
  std::string compute_dtype;

 public:
  explicit RequantizeConfigNode(bool is_default = false) : is_default(is_default) {}

  std::string get_rounding() const {
    if (!rounding.empty()) return rounding;
    return "UPWARD";
  }

  std::string get_compute_dtype() const {
    if (!compute_dtype.empty()) return compute_dtype;

    // For the x86 architecture, the float32 computation is expected to give significant speedup,
    // with little loss in the accuracy of the requantize operation.
    auto target = Target::Current(true);
    auto target_has_feature_fn_ptr = tvm::runtime::Registry::Get("target.target_has_feature");
    ICHECK(target_has_feature_fn_ptr) << "Function target.target_has_feature not found";
    if (target.defined() && target->kind->name == "llvm") {
      if ((*target_has_feature_fn_ptr)("sse4.1", target)) {
        return "float32";
      }
    }
    return "int64";
  }

  const bool is_default = false;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("rounding", &rounding);
    v->Visit("compute_dtype", &compute_dtype);
  }

  static constexpr const char* _type_key = "relay.qnn.op.RequantizeConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(RequantizeConfigNode, Object);
};

/*!
 * \brief Container for build configuration options
 */
class RequantizeConfig : public ObjectRef {
 public:
  RequantizeConfig() {}
  explicit RequantizeConfig(ObjectPtr<Object> n) : ObjectRef(n) {}

  const RequantizeConfigNode* operator->() const {
    return static_cast<const RequantizeConfigNode*>(get());
  }

  RequantizeConfigNode* operator->() { return static_cast<RequantizeConfigNode*>(get_mutable()); }

  /*!
   * \brief Push a new BuildConfig context onto the thread local stack.
   * \param build_config The configuration to set as the current context.
   */
  static void EnterRequantizeConfigScope(const RequantizeConfig& requantize_config);

  /*!
   * \brief Pop a build config off the thread local context stack, restoring the previous
   * configuration as the current context.
   */
  static void ExitRequantizeConfigScope();

  /*!
   * \brief Get the current BuildConfig context from thread local storage, or a default
   * configuration if a BuildConfig scope has not been entered.
   * \return The configuration that is the current context.
   */
  static RequantizeConfig& Current();

  using ContainerType = RequantizeConfigNode;
};

}  // namespace qnn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_QNN_OP_REQUANTIZE_CONFIG_H_
