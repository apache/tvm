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
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace codegen {
namespace intrin {
using tir::FLowerIntrinsic;

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                                     DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                                     DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                                     DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.log1p")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.tan").set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                                     DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.atan")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.atanh")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.atan2")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                                     DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.acos")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.acosh")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                                     DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.asin")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.asinh")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.hypot")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.nextafter")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.copysign")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.ldexp")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                                     DispatchPureExtern<FloatSuffix>);

}  // namespace intrin

namespace legalize {

using namespace tir;

TVM_REGISTER_OP("tir.rsqrt")
    .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const CallNode* call = e.as<CallNode>();
      ICHECK(call != nullptr);
      auto one = make_const(call->args[0].dtype(), 1);
      return one / sqrt(call->args[0]);
    });

TVM_REGISTER_OP("tir.sigmoid")
    .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const CallNode* call = e.as<CallNode>();
      ICHECK(call != nullptr);
      auto one = make_const(call->args[0].dtype(), 1);
      return one / (one + exp(-call->args[0]));
    });

TVM_REGISTER_OP("tir.isfinite")
    .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const CallNode* call = e.as<CallNode>();
      ICHECK(call != nullptr);
      return isfinite(call->args[0]);
    });

TVM_REGISTER_OP("tir.isinf")
    .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const CallNode* call = e.as<CallNode>();
      ICHECK(call != nullptr);
      return isinf(call->args[0]);
    });

/*!
 * \brief Makes fixed point multiplication.
 * \param x Input tensor.
 * \param y Integer multiplier.
 * \param left_shift Integer left shift.
 * \param right_shift Integer right shift.
 * \param is_left_shift_required Flag whether we need to do left shift or not.
 * \return Calculated expression.
 */
static PrimExpr QMultiplyShift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr left_shift,
                               PrimExpr right_shift, PrimExpr is_left_shift_required) {
  // Only int32 types are supported (any number of lanes is allowed)
  ICHECK(y.dtype().code() == DLDataTypeCode::kDLInt && y.dtype().bits() == 32);
  ICHECK(left_shift.dtype().code() == DLDataTypeCode::kDLInt && left_shift.dtype().bits() == 32);
  ICHECK(right_shift.dtype().code() == DLDataTypeCode::kDLInt && right_shift.dtype().bits() == 32);

  DataType hp_dtype = DataType::Int(64, x.dtype().lanes());
  DataType lp_dtype = DataType::Int(32, x.dtype().lanes());

  // 1) Cast and Multiply the integer multiplier
  PrimExpr one = make_const(hp_dtype, 1);
  x = cast(hp_dtype, x);
  y = cast(hp_dtype, y);
  x = tir::Select(is_left_shift_required, x << left_shift, x);

  // 2) Perform the multiplication in higher precision.
  x = x * y;

  // 3) Find the rounding scalar
  PrimExpr total_right_shift = right_shift + q;
  PrimExpr pos_rounding_value = (one << (total_right_shift - 1));
  x = x + pos_rounding_value;

  // 4) Simply right shift the result to get the final output.
  x = x >> total_right_shift;

  // 5) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return cast(lp_dtype, x);
}

TVM_REGISTER_OP("tir.q_multiply_shift")
    .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;

      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);

      PrimExpr x = call->args[0];
      PrimExpr y = call->args[1];
      PrimExpr q = call->args[2];
      PrimExpr s = call->args[3];

      // Lambda function to extract the int value from PrimExpr
      auto get_int_value = [](const PrimExpr node) {
        if (auto int_node = node.as<IntImmNode>()) {
          return int_node->value;
        }
        auto broadcast_node = node.as<BroadcastNode>();
        CHECK(broadcast_node != nullptr);
        auto int_node = broadcast_node->value.as<IntImmNode>();
        CHECK(int_node != nullptr);
        return int_node->value;
      };
      // Power of 2 is determined by the fixed_point_multiplier == 1 << 30. In case of power of
      // 2, fixed point multiplier will represent a float value of 0.5. In fixed point, this is
      // represented by 1 << 30.
      if (get_int_value(y) == (1 << 30)) {
        PrimExpr exp = s - 1;
        int exp_val = get_int_value(s) - 1;
        if (exp_val > 0) {
          // power of 2 is greater than 0, apply left shift.
          return x << exp;
        } else {
          // power of 2 is less than 0, round and then apply right shift.
          DataType lp_dtype = DataType::Int(32, x.dtype().lanes());
          PrimExpr one = make_const(lp_dtype, 1);
          exp = -exp;
          PrimExpr rounding_factor = one << (exp - 1);
          PrimExpr rounded_t = x + rounding_factor;
          return rounded_t >> exp;
        }
      } else {
        // Only int32 types are supported (any number of lanes is allowed)
        ICHECK(s.dtype().code() == DLDataTypeCode::kDLInt && s.dtype().bits() == 32);

        // Calculating integer shifts
        PrimExpr zero = make_const(s.dtype(), 0);
        PrimExpr left_shift = tir::Select(s > zero, s, zero);
        PrimExpr right_shift = tir::Select(s > zero, zero, -s);
        PrimExpr is_left_shift_required = (left_shift != zero);

        return QMultiplyShift(x, y, q, left_shift, right_shift, is_left_shift_required);
      }
    });

TVM_REGISTER_OP("tir.q_multiply_shift_per_axis")
    .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);

      PrimExpr x = call->args[0];
      PrimExpr y = call->args[1];
      PrimExpr left_shift = call->args[2];
      PrimExpr right_shift = call->args[3];
      PrimExpr q = call->args[4];
      PrimExpr is_lshift_required = call->args[5];
      // Note, 7th argument is "is_rshift_required" flag, but we don't need that here.
      // PrimExpr is_rshift_required = call->args[6];

      return QMultiplyShift(x, y, q, left_shift, right_shift, is_lshift_required);
    });
}  // namespace legalize
}  // namespace codegen
}  // namespace tvm
