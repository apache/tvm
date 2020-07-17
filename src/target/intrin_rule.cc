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
#include "intrin_rule.h"

#include <tvm/tir/op.h>

namespace tvm {
namespace codegen {
namespace intrin {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.exp").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.erf").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log2").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log10").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.log1p").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.tanh").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.tan").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.atan").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.atanh").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.atan2").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.cos").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.acos").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.cosh").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.acosh").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sin").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.asin").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sinh").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.asinh").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.hypot").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.nextafter").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.copysign").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.ldexp").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sqrt").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.rsqrt")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      PrimExpr e = args[0];
      const CallNode* call = e.as<CallNode>();
      CHECK(call != nullptr);

      auto one = make_const(call->args[0].dtype(), 1);
      *rv = one / sqrt(call->args[0]);
    });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.pow").set_body(DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.sigmoid")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      PrimExpr e = args[0];
      const CallNode* call = e.as<CallNode>();
      CHECK(call != nullptr);

      auto one = make_const(call->args[0].dtype(), 1);
      *rv = one / (one + exp(-call->args[0]));
    });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.isfinite")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      PrimExpr e = args[0];
      const CallNode* call = e.as<CallNode>();
      CHECK(call != nullptr);
      *rv = isfinite(call->args[0]);
    });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.isinf")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      PrimExpr e = args[0];
      const CallNode* call = e.as<CallNode>();
      CHECK(call != nullptr);
      *rv = isinf(call->args[0]);
    });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.q_multiply_shift")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      using tir::make_const;

      PrimExpr e = args[0];
      const tir::CallNode* call = e.as<tir::CallNode>();
      CHECK(call != nullptr);

      PrimExpr x = call->args[0];
      PrimExpr y = call->args[1];
      PrimExpr q = call->args[2];
      PrimExpr s = call->args[3];

      // Only int32 types are supported (any number of lanes is allowed)
      CHECK(y.dtype().code() == DLDataTypeCode::kDLInt && y.dtype().bits() == 32);
      CHECK(s.dtype().code() == DLDataTypeCode::kDLInt && s.dtype().bits() == 32);

      DataType hp_dtype = DataType::Int(64, x.dtype().lanes());
      DataType lp_dtype = DataType::Int(32, x.dtype().lanes());

      // 1) Calculating the integer multiplier and integer shift
      PrimExpr zero = make_const(s.dtype(), 0);
      PrimExpr left_shift = tir::Select(s > zero, s, zero);
      PrimExpr right_shift = tir::Select(s > zero, zero, -s);

      // 2) Cast and Multiply the integer multiplier
      PrimExpr one = make_const(hp_dtype, 1);
      x = cast(hp_dtype, x);
      y = cast(hp_dtype, y);
      x = tir::Select(left_shift != zero, x << left_shift, x);

      // 3) Perform the multiplication in higher precision.
      x = x * y;

      // 4) Find the rounding scalar
      PrimExpr total_right_shift = right_shift + q;
      PrimExpr pos_rounding_value = (one << (total_right_shift - 1));
      x = x + pos_rounding_value;

      // 5) Simply right shift the result to get the final output.
      x = x >> total_right_shift;

      // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
      *rv = cast(lp_dtype, x);
    });

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
