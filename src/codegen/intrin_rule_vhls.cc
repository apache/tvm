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
 *  Copyright (c) 2018 by Contributors
 * \file intrin_rule_vhls.cc
 * \brief VHLS intrinsic rules.
 */
#include "intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.floor")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.ceil")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.trunc")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.fabs")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.round")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.sdaccel.popcount")
.set_body(DispatchExtern<Direct>);


}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
