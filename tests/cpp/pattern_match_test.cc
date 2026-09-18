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

#include "../src/sym/pattern_match.h"

#include <gtest/gtest.h>
#include <tvm/ir/prim/expr.h>

TEST(Pattern, Basic) {
  using namespace tvm;
  using namespace tvm::sym;
  tvm::PrimVar x("x"), y("y"), z("z");
  PrimExpr scalable_lanes = prim::Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}), 4);
  sym::PVar<PrimExpr> px, py, pz;
  sym::PVar<DLDataType> pt;
  sym::PVar<PrimExpr> planes;

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

    TVM_FFI_ICHECK(prim::ExprDeepEqual()(rr, 1 + y));
    TVM_FFI_ICHECK(prim::ExprDeepEqual()(px.Eval() + py.Eval(), 1 + y));
  }
  {
    TVM_FFI_ICHECK((px + max(py, px)).Match((x + 1) + max(y, (x + 1))));
    TVM_FFI_ICHECK(prim::ExprDeepEqual()(px.Eval(), x + 1));
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
    TVM_FFI_ICHECK(select(px >= pz, py, py + pz).Match(prim::Select((x + 1) >= 1, y, y + 1)));
    TVM_FFI_ICHECK(prim::ExprDeepEqual()(px.Eval(), x + 1));
  }
  // bit intrinsics
  {
    TVM_FFI_ICHECK((px >> pz).Match(x >> 1));
    TVM_FFI_ICHECK(prim::is_const_int(pz.Eval(), 1));
  }
  TVM_FFI_ICHECK(!(px >> pz).Match(x << 1));
  TVM_FFI_ICHECK((px << pz).Match(x << 1));
  TVM_FFI_ICHECK((px & pz).Match(x & 1));
  TVM_FFI_ICHECK((px | pz).Match(x | 1));
  TVM_FFI_ICHECK((px ^ pz).Match(x ^ 1));
  TVM_FFI_ICHECK((px - (~(py | (px * pz)))).Match(x - (~(2 | (x * 2)))));
  // select
  {
    TVM_FFI_ICHECK(select(px > pz, py, py + pz).Match(prim::Select(x > 1, y, y + 1)));
    TVM_FFI_ICHECK(prim::is_const_int(pz.Eval(), 1));
  }
  TVM_FFI_ICHECK(!select(px > pz, py, py + pz).Match(prim::Select(x > 2, y, y + 1)));
  TVM_FFI_ICHECK(!select(px > pz, py, py).Match(prim::Select(x > 2, y, y + 1)));
  {
    TVM_FFI_ICHECK(select(px, py, pz).Match(prim::Select(x > 2, y, y + 1)));
    TVM_FFI_ICHECK(prim::ExprDeepEqual()(pz.Eval(), y + 1));
  }
  // if_then_else
  {
    TVM_FFI_ICHECK(if_then_else(px > pz, py, py + pz).Match(if_then_else(x > 1, y, y + 1)));
    TVM_FFI_ICHECK(prim::is_const_int(pz.Eval(), 1));
  }
  // cast pattern
  {
    TVM_FFI_ICHECK(!cast(PConst<DLDataType>(DLDataType{kDLInt, 32, 1}), px)
                        .Match(prim::Cast(PrimType::Float(64), x)));
    TVM_FFI_ICHECK(cast(pt, px).Match(prim::Cast(PrimType::Float(64), x)));
    TVM_FFI_ICHECK((pt.Eval() == DLDataType{kDLFloat, 64, 1}));
    auto zz = cast(pt, px).Eval();
    TVM_FFI_ICHECK(
        (cast(pt, px) - cast(pt, py))
            .Match(prim::Cast(PrimType::Float(64), x) - prim::Cast(PrimType::Int(64), x)));
    auto expr = prim::Cast(PrimType::Int(32), prim::Cast(PrimType::Float(64), x));
    TVM_FFI_ICHECK(!(cast(pt, cast(pt, px))).Match(expr));
  }
  // ramp pattern
  {
    TVM_FFI_ICHECK(ramp(px, PConst<PrimExpr>(1), planes).Match(prim::Ramp(x, 1, 10)));
    TVM_FFI_ICHECK(planes.Eval().as<prim::IntImmNode>()->value == 10);
    TVM_FFI_ICHECK(ramp(px, PConst<PrimExpr>(1), planes).Match(prim::Ramp(x, 1, scalable_lanes)));
    TVM_FFI_ICHECK(prim::ExprDeepEqual()(planes.Eval(), scalable_lanes));
    TVM_FFI_ICHECK(!ramp(px, PConst<PrimExpr>(1), planes).Match(prim::Ramp(x, 2, 10)));
  }
  // broadcast pattern
  {
    TVM_FFI_ICHECK(broadcast(px, planes).Match(prim::Broadcast(x, 10)));
    TVM_FFI_ICHECK(planes.Eval().as<prim::IntImmNode>()->value == 10);
    TVM_FFI_ICHECK(broadcast(px * py, planes).Match(prim::Broadcast(x * 10, 10)));
    TVM_FFI_ICHECK(broadcast(px, planes).Match(prim::Broadcast(x, scalable_lanes)));
    TVM_FFI_ICHECK(prim::ExprDeepEqual()(planes.Eval(), scalable_lanes));
  }
}

TEST(Pattern, IntImm) {
  using namespace tvm;
  PrimVar tx("tx"), ty("ty");
  sym::PVar<prim::IntImm> c;
  sym::PVar<Var> v;
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
  sym::PVarWithDataType<PrimExpr, sym::PConst<DLDataType>> pat(DLDataType{kDLFloat, 32, 1});
  PrimVar x("x", PrimType::Float(32));
  PrimVar y("y", PrimType::Float(32));
  PrimVar x_int("x", PrimType::Int(32));
  PrimVar y_int("y", PrimType::Int(32));
  TVM_FFI_ICHECK(pat.Match(x + y * 2.0f));
  TVM_FFI_ICHECK(!pat.Match(x_int + y_int * 2));

  // match vectorized expr with specified element dtype
  sym::PVecDataType vec_ty(DLDataType{kDLFloat, 32, 1});
  sym::PVarWithDataType<PrimExpr, sym::PVecDataType> vpat(vec_ty);
  PrimVar vx("x", PrimType::Float(32, 8));
  PrimVar vy("y", PrimType::Float(32, 8));
  PrimVar vx_int("x", PrimType::Int(32, 8));
  PrimVar vy_int("y", PrimType::Int(32, 8));
  TVM_FFI_ICHECK(vpat.Match(vx + vy * prim::Broadcast(2.0f, 8)));
  TVM_FFI_ICHECK(!vpat.Match(vx_int + vy_int * prim::Broadcast(2, 8)));
}
