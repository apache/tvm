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
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_default.cc
 * \brief Default intrinsic rules.
 */
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.exp")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.tanh")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sqrt")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.rsqrt")
.set_body([](const TVMArgs& args, TVMRetValue* rv){
    Expr e = args[0];
    const Call* call = e.as<Call>();
    CHECK(call != nullptr);

    auto one = make_const(call->args[0].type(), 1);
    *rv = one / sqrt(call->args[0]);
  });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.pow")
.set_body(DispatchExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sigmoid")
.set_body([](const TVMArgs& args, TVMRetValue* rv){
    Expr e = args[0];
    const Call* call = e.as<Call>();
    CHECK(call != nullptr);

    auto one = make_const(call->args[0].type(), 1);
    *rv = one / (one + exp(-call->args[0]));
  });

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
