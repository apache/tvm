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

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

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
  std::string operator()(DataType t, std::string name) const { return name; }
};

// Call pure extern function.
template <typename T>
inline PrimExpr DispatchPureExtern(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  // Use string based dispatch to extern for backward compact
  // TODO(tvm-team) replace once the new dispatching system is inplace.
  const OpNode* op = call->op.as<OpNode>();
  ICHECK(op != nullptr);
  std::string name = op->name;
  ICHECK_EQ(name.substr(0, 4), "tir.");
  name = T()(call->dtype, name.substr(4));

  if (name.length() != 0) {
    Array<PrimExpr> new_args = {StringImm(name)};
    for (auto arg : call->args) {
      new_args.push_back(arg);
    }
    return Call(call->dtype, builtin::call_pure_extern(), new_args);
  } else {
    return e;
  }
}

// Dispatch ERF to fast erf when it is not available.
PrimExpr DispatchFastErf(const PrimExpr& e);

// Dispatch numerically stable tanh such that tanh(large_num) does not result in NaN
PrimExpr DispatchNumericalStableTanh(const PrimExpr& e);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_INTRIN_RULE_H_
