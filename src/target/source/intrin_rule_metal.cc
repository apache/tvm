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
#include <tvm/tir/op_attr_types.h>
#include <tvm/topi/elemwise.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
using tir::FLowerIntrinsic;

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp10")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fmod")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

// There is no erf function in Metal. When erf is used, we use fast_erf instead
static PrimExpr DispatchFastErf(const PrimExpr& e) {
  LOG(WARNING) << " Metal doesn't have built-in erf function. fast_erf will be used instead.";
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1);
  PrimExpr arg = call->args[0];
  int bits = arg.dtype().bits();
  bool isFloat = arg.dtype().is_float();
  PrimExpr res;
  if (isFloat && (bits == 16 || bits == 32))
    res = topi::fast_erf_float_expr(arg, bits);
  else
    LOG(FATAL) << "Unsupported type in Metal fast_erf";
  return res;
}
TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchFastErf);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
