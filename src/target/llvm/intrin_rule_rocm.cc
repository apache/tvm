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
 * \file intrin_rule_rocm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/tir/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/runtime/registry.h>

#include <sstream>

namespace tvm {
namespace codegen {

inline void DispatchExternOCML(const TVMArgs& args, TVMRetValue* rv) {
  PrimExpr e = args[0];
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  CHECK(call != nullptr);
  std::ostringstream intrinsic_name;
  intrinsic_name << "__ocml_" << call->name << "_f" << call->dtype.bits();
  *rv = CallNode::make(call->dtype, intrinsic_name.str(), call->args,
                   CallNode::PureExtern);
}

namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.floor")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.ceil")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.round")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.trunc")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fabs")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.exp")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.erf")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fma")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.log")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sqrt")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.pow")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tanh")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tan")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.cos")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sin")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.atan")
.set_body(DispatchExternOCML);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
