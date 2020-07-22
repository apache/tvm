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
 * \file intrin_rule_aocl.cc
 * \brief AOCL intrinsic rules.
 */
#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.floor").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.ceil").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.trunc").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.fabs").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.round").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.exp").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.log").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.tanh").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.sqrt").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.pow").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl.popcount").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.floor").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.ceil").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.trunc").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.fabs").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.round").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.exp").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.log").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.tanh").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.sqrt").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.pow").set_body(DispatchPureExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.aocl_sw_emu.popcount").set_body(DispatchPureExtern<Direct>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
