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
 * \file intrin_rule_metal.cc
 * \brief Metal intrinsic rules.
 */
#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.floor").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.ceil").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.trunc").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.fabs").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.round").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp2").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.exp10").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log2").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.log10").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.tanh").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sqrt").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.pow").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.popcount").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.fmod").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sin").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.sinh").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.cos").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.metal.cosh").set_body(DispatchPureExtern<Direct>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
