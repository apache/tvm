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

#include "../src/arith/pattern_match.h"

#include <gtest/gtest.h>
#include <tvm/tir/analysis.h>

TEST(Pattern, Basic) {
  using namespace tvm;
  using namespace tvm::tir;
  using namespace tvm::arith;
  tvm::tir::Var x("x"), y("y"), z("z");
  PrimExpr scalable_lanes = Mul(Call(DataType::Int(32), builtin::vscale(), {}), 4);
  arith::PVar<PrimExpr> px, py, pz;
  arith::PVar<DataType> pt;
  arith::PVar<PrimExpr> planes;
  arith::PCallExpr<PVscaleOp> vscale;

  // arithmetics
  auto r = 1 + (y + 1);
  TVM_FFI_ICHECK(!(px + (px + px)).Match(r));
  TVM_FFI_ICHECK(!(px + (py + py)).Match(r));
  TVM_FFI_ICHECK((px + (py + pz)).Match(r));
  auto pattern = px + (py + pz);
  TVM_FFI_ICHECK(pattern.Match(r));
  {
    TVM_FFI_ICHECK((px + (py + px)).Match(r));
    auto rr = (px + py).Eval();

    TVM_FFI_ICHECK(tir::ExprDeepEqual()(rr, 1 + y));
    TVM_FFI_ICHECK(tir::ExprDeepEqual()(px.Eval() + py.Eval(), 1 + y));
  }
  {
    TVM_FFI_ICHECK((px + max(py, px)).Match((x + 1) + max(y, (x + 1))));
    TVM_FFI_ICHECK(tir::ExprDeepEqual()(px.Eval(), x + 1));
  }
  TVM_FFI_ICHECK(!(px + min(py, px)).Match((x + 1) + max(y, (x + 1))));

  TVM_FFI_ICHECK((px + min(py, px)).Match(z + min(y, z)));
  TVM_FFI_ICHECK((px + truncdiv(py, px * py)).Match(x + truncdiv(2, x * 2)));
  TVM_FFI_ICHECK((px - truncmod(py, px * pz)).Match(x - truncmod(2, x * 2)));
  TVM_FFI_ICHECK((px - floormod(py, px * PConst<PrimExpr>(2))).Match(x - floormod(2, x * 2)));

  // logicals
  TVM_FFI_ICHECK((px == pz).Match(x == 1));
  TVM_FFI_ICHECK((px != pz).Match(x != 1));
  TVM_FFI_ICHECK((px > py).Match(x > y));
  TVM_FFI_ICHECK((px < py).Match(x < y));
  TVM_FFI_ICHECK((px <= py).Match(x <= y));
  TVM_FFI_ICHECK((px >= py).Match(x >= y));
  TVM_FFI_ICHECK((px >= py && px < pz).Match(x >= y && x < z));
  TVM_FFI_ICHECK((!(px > py || px != py)).Match(!(x > y || x != y)));
  {
    TVM_FFI_ICHECK(select(px >= pz, py, py + pz).Match(tir::Select((x + 1) >= 1, y, y + 1)));
    TVM_FFI_ICHECK(tir::ExprDeepEqual()(px.Eval(), x + 1));
  }
  // bit intrinsics
  {
    TVM_FFI_ICHECK((px >> pz).Match(x >> 1));
    TVM_FFI_ICHECK(is_const_int(pz.Eval(), 1));
  }
  TVM_FFI_ICHECK(!(px >> pz).Match(x << 1));
  TVM_FFI_ICHECK((px << pz).Match(x << 1));
  TVM_FFI_ICHECK((px & pz).Match(x & 1));
  TVM_FFI_ICHECK((px | pz).Match(x | 1));
  TVM_FFI_ICHECK((px ^ pz).Match(x ^ 1));
  TVM_FFI_ICHECK((px - (~(py | (px * pz)))).Match(x - (~(2 | (x * 2)))));
  // select
  {
    TVM_FFI_ICHECK(select(px > pz, py, py + pz).Match(tir::Select(x > 1, y, y + 1)));
    TVM_FFI_ICHECK(is_const_int(pz.Eval(), 1));
  }
  TVM_FFI_ICHECK(!select(px > pz, py, py + pz).Match(tir::Select(x > 2, y, y + 1)));
  TVM_FFI_ICHECK(!select(px > pz, py, py).Match(tir::Select(x > 2, y, y + 1)));
  {
    TVM_FFI_ICHECK(select(px, py, pz).Match(tir::Select(x > 2, y, y + 1)));
    TVM_FFI_ICHECK(tir::ExprDeepEqual()(pz.Eval(), y + 1));
  }
  // if_then_else
  {
    TVM_FFI_ICHECK(if_then_else(px > pz, py, py + pz).Match(if_then_else(x > 1, y, y + 1)));
    TVM_FFI_ICHECK(is_const_int(pz.Eval(), 1));
  }
  // cast pattern
  {
    TVM_FFI_ICHECK(
        !cast(PConst<DataType>(DataType::Int(32)), px).Match(tir::Cast(DataType::Float(64), x)));
    TVM_FFI_ICHECK(cast(pt, px).Match(tir::Cast(DataType::Float(64), x)));
    TVM_FFI_ICHECK(pt.Eval() == DataType::Float(64));
    auto zz = cast(pt, px).Eval();
    TVM_FFI_ICHECK((cast(pt, px) - cast(pt, py))
                       .Match(tir::Cast(DataType::Float(64), x) - tir::Cast(DataType::Int(64), x)));
    auto expr = tir::Cast(DataType::Int(32), tir::Cast(DataType::Float(64), x));
    TVM_FFI_ICHECK(!(cast(pt, cast(pt, px))).Match(expr));
  }
  // ramp pattern
  {
    TVM_FFI_ICHECK(ramp(px, PConst<PrimExpr>(1), planes).Match(tir::Ramp(x, 1, 10)));
    TVM_FFI_ICHECK(planes.Eval().as<IntImmNode>()->value == 10);
    TVM_FFI_ICHECK(ramp(px, PConst<PrimExpr>(1), planes).Match(tir::Ramp(x, 1, scalable_lanes)));
    TVM_FFI_ICHECK((vscale * PConst<PrimExpr>(4)).Match(planes.Eval()));
    TVM_FFI_ICHECK(!ramp(px, PConst<PrimExpr>(1), planes).Match(tir::Ramp(x, 2, 10)));
  }
  // broadcast pattern
  {
    TVM_FFI_ICHECK(broadcast(px, planes).Match(tir::Broadcast(x, 10)));
    TVM_FFI_ICHECK(planes.Eval().as<IntImmNode>()->value == 10);
    TVM_FFI_ICHECK(broadcast(px * py, planes).Match(tir::Broadcast(x * 10, 10)));
    TVM_FFI_ICHECK(broadcast(px, planes).Match(tir::Broadcast(x, scalable_lanes)));
    TVM_FFI_ICHECK((vscale * PConst<PrimExpr>(4)).Match(planes.Eval()));
  }
}

TEST(Pattern, IntImm) {
  using namespace tvm;
  tir::Var tx, ty;
  arith::PVar<IntImm> c;
  arith::PVar<tir::Var> v;
  {
    // We can match integer and Var, both of which are
    // special case container of Expr
    TVM_FFI_ICHECK((v * c).Match(tx * 3));
    TVM_FFI_ICHECK_EQ(c.Eval()->value, 3);
    TVM_FFI_ICHECK((v * 3).Match(tx * 3));
  }
  // cannot match c to ty
  TVM_FFI_ICHECK(!(v * c).Match(tx * ty));
  // cannot match tx + 1 to v
  TVM_FFI_ICHECK(!(v * c).Match((tx + 1) * 3));
}

TEST(Pattern, MatchWithType) {
  using namespace tvm;
  // match expr with specified dtype
  arith::PVarWithDataType<PrimExpr, arith::PConst<DataType>> pat(DataType::Float(32));
  tir::Var x("x", DataType::Float(32));
  tir::Var y("y", DataType::Float(32));
  tir::Var x_int("x", DataType::Int(32));
  tir::Var y_int("y", DataType::Int(32));
  TVM_FFI_ICHECK(pat.Match(x + y * 2.0f));
  TVM_FFI_ICHECK(!pat.Match(x_int + y_int * 2));

  // match vectorized expr with specified element dtype
  arith::PVecDataType vec_ty(DataType::Float(32));
  arith::PVarWithDataType<PrimExpr, arith::PVecDataType> vpat(vec_ty);
  tir::Var vx = tir::Var("x", DataType::Float(32, 8));
  tir::Var vy("y", DataType::Float(32, 8));
  tir::Var vx_int("x", DataType::Int(32, 8));
  tir::Var vy_int("y", DataType::Int(32, 8));
  TVM_FFI_ICHECK(vpat.Match(vx + vy * tir::Broadcast(2.0f, 8)));
  TVM_FFI_ICHECK(!vpat.Match(vx_int + vy_int * tir::Broadcast(2, 8)));
}
