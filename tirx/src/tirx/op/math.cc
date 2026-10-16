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
 * \file tirx/op/math.cc
 * \brief Specialized mathematical expression builders.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/math.h>

#include "../../ir/prim/op_utils.h"

namespace tvm {
namespace tirx {

using namespace prim;
using namespace prim::detail;

PrimExpr logaddexp(PrimExpr a, PrimExpr b, Location loc) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << b;
  BinaryOpMatchTypes(a, b, loc);
  PrimExpr exp_sum = add(exp(a), exp(b));
  PrimExpr log_exp_sum = log(exp_sum);
  return log_exp_sum;
}

PrimExpr fast_erf_float_expr(PrimExpr arg, int bits) {
  PrimType fp_ty = PrimType::Float(bits);
  auto plus_4 = FloatImm(fp_ty, 4.f);
  auto minus_4 = FloatImm(fp_ty, -4.f);

  // The monomial coefficients of the numerator polynomial (odd).
  auto alpha_1 = FloatImm(fp_ty, -1.60960333262415e-02f);
  auto alpha_3 = FloatImm(fp_ty, -2.95459980854025e-03f);
  auto alpha_5 = FloatImm(fp_ty, -7.34990630326855e-04f);
  auto alpha_7 = FloatImm(fp_ty, -5.69250639462346e-05f);
  auto alpha_9 = FloatImm(fp_ty, -2.10102402082508e-06f);
  auto alpha_11 = FloatImm(fp_ty, 2.77068142495902e-08f);
  auto alpha_13 = FloatImm(fp_ty, -2.72614225801306e-10f);

  // The monomial coefficients of the denominator polynomial (even).
  auto beta_0 = FloatImm(fp_ty, -1.42647390514189e-02f);
  auto beta_2 = FloatImm(fp_ty, -7.37332916720468e-03f);
  auto beta_4 = FloatImm(fp_ty, -1.68282697438203e-03f);
  auto beta_6 = FloatImm(fp_ty, -2.13374055278905e-04f);
  auto beta_8 = FloatImm(fp_ty, -1.45660718464996e-05f);

  // clamp x
  auto x = tvm::max(tvm::min(arg, plus_4), minus_4);
  auto x2 = x * x;

  // Evaluate the numerator polynomial p.
  auto p = x2 * alpha_13 + alpha_11;
  p = x2 * p + alpha_9;
  p = x2 * p + alpha_7;
  p = x2 * p + alpha_5;
  p = x2 * p + alpha_3;
  p = x2 * p + alpha_1;
  p = x * p;

  // Evaluate the denominator polynomial p.
  auto q = x2 * beta_8 + beta_6;
  q = x2 * q + beta_4;
  q = x2 * q + beta_2;
  q = x2 * q + beta_0;

  return p / q;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tirx._OpLogAddExp", [](PrimExpr a, PrimExpr b, Location loc) {
    return logaddexp(a, b, loc);
  });
}

}  // namespace tirx
}  // namespace tvm
