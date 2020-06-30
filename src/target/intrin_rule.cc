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

TVM_REGISTER_GLOBAL("tvm.intrin.rule.default.fixed_point_multiply")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      using tir::make_const;

      PrimExpr e = args[0];
      const tir::CallNode* call = e.as<tir::CallNode>();
      CHECK(call != nullptr);

      PrimExpr tensor = call->args[0];
      PrimExpr fixed_point_multiplier = call->args[1];
      PrimExpr shift = call->args[2];

      // Only int32 types are supported (any number of lanes is allowed)
      CHECK(tensor.dtype().code() == DLDataTypeCode::kDLInt && tensor.dtype().bits() == 32);
      CHECK(fixed_point_multiplier.dtype().code() == DLDataTypeCode::kDLInt &&
            fixed_point_multiplier.dtype().bits() == 32);
      CHECK(shift.dtype().code() == DLDataTypeCode::kDLInt && shift.dtype().bits() == 32);

      DataType hp_dtype = DataType::Int(64, tensor.dtype().lanes());
      DataType lp_dtype = DataType::Int(32, tensor.dtype().lanes());

      // 1) Calculating the integer multiplier and integer shift
      PrimExpr zero = make_const(shift.dtype(), 0);
      PrimExpr left_shift = tir::Select((shift > zero), shift, zero);
      PrimExpr right_shift = tir::Select(shift > zero, zero, -shift);

      // 2) Multiply the integer multiplier
      tensor = tir::Select(left_shift != zero, tensor << cast(hp_dtype, left_shift),
                           cast(hp_dtype, tensor));

      // 3) Perform the multiplication in higher precision.
      tensor = tensor * fixed_point_multiplier;

      // 4) Find the rounding scalar
      PrimExpr total_right_shift = right_shift + 31;
      PrimExpr pos_rounding_value = (make_const(hp_dtype, 1) << (total_right_shift - 1));

      tensor = tensor + pos_rounding_value;

      // 5) Simply right shift the result to get the final output.
      tensor = tensor >> total_right_shift;

      // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
      *rv = cast(lp_dtype, tensor);
    });

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
