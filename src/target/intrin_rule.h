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
 * \file intrin_rule.h
 * \brief Utility to generate intrinsic rules
 */
#ifndef TVM_TARGET_INTRIN_RULE_H_
#define TVM_TARGET_INTRIN_RULE_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/runtime/registry.h>
#include <string>

namespace tvm {
namespace codegen {
namespace intrin {
using namespace tir;

// Add float suffix to the intrinsics
struct FloatSuffix {
  std::string operator()(DataType t, std::string name) const {
    if (t == DataType::Float(32)) {
      return name + 'f';
    } else if (t == DataType::Float(64)) {
      return name;
    } else {
      return "";
    }
  }
};

// Return the intrinsic name
struct Direct {
  std::string operator()(DataType t, std::string name) const {
    return name;
  }
};

// Call pure extern function.
template<typename T>
inline void DispatchExtern(const TVMArgs& args, TVMRetValue* rv) {
  PrimExpr e = args[0];
  const CallNode* call = e.as<CallNode>();
  CHECK(call != nullptr);
  std::string name = T()(call->dtype, call->name);
  if (name.length() != 0) {
    *rv = CallNode::make(
        call->dtype, name, call->args, CallNode::PureExtern);
  } else {
    *rv = e;
  }
}

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_INTRIN_RULE_H_
