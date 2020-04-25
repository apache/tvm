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
 * \file intrin_rule_default.cc
 * \brief Default intrinsic rules.
 */
#include <tvm/tir/op.h>
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.exp")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.erf")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log2")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log10")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log1p")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.tanh")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.tan")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.cos")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.cosh")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sin")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sinh")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.atan")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.atan2")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.hypot")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.nextafter")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.copysign")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.ldexp")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sqrt")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.rsqrt")
.set_body([](const TVMArgs& args, TVMRetValue* rv){
    PrimExpr e = args[0];
    const CallNode* call = e.as<CallNode>();
    CHECK(call != nullptr);

    auto one = make_const(call->args[0].dtype(), 1);
    *rv = one / sqrt(call->args[0]);
  });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.pow")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sigmoid")
.set_body([](const TVMArgs& args, TVMRetValue* rv){
    PrimExpr e = args[0];
    const CallNode* call = e.as<CallNode>();
    CHECK(call != nullptr);

    auto one = make_const(call->args[0].dtype(), 1);
    *rv = one / (one + exp(-call->args[0]));
  });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.isfinite")
.set_body([](const TVMArgs& args, TVMRetValue* rv){
    PrimExpr e = args[0];
    const CallNode* call = e.as<CallNode>();
    CHECK(call != nullptr);
    *rv = isfinite(call->args[0]);
  });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.isinf")
.set_body([](const TVMArgs& args, TVMRetValue* rv){
    PrimExpr e = args[0];
    const CallNode* call = e.as<CallNode>();
    CHECK(call != nullptr);
    *rv = isinf(call->args[0]);
  });

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
