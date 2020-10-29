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
  arith::PVar<PrimExpr> px, py, pz;
  arith::PVar<DataType> pt;
  arith::PVar<int> planes;

  // arithmetics
  auto r = 1 + (y + 1);
  ICHECK(!(px + (px + px)).Match(r));
  ICHECK(!(px + (py + py)).Match(r));
  ICHECK((px + (py + pz)).Match(r));
  auto pattern = px + (py + pz);
  ICHECK(pattern.Match(r));
  {
    ICHECK((px + (py + px)).Match(r));
    auto rr = (px + py).Eval();

    ICHECK(tir::ExprDeepEqual()(rr, 1 + y));
    ICHECK(tir::ExprDeepEqual()(px.Eval() + py.Eval(), 1 + y));
  }
  {
    ICHECK((px + max(py, px)).Match((x + 1) + max(y, (x + 1))));
    ICHECK(tir::ExprDeepEqual()(px.Eval(), x + 1));
  }
  ICHECK(!(px + min(py, px)).Match((x + 1) + max(y, (x + 1))));

  ICHECK((px + min(py, px)).Match(z + min(y, z)));
  ICHECK((px + truncdiv(py, px * py)).Match(x + truncdiv(2, x * 2)));
  ICHECK((px - truncmod(py, px * pz)).Match(x - truncmod(2, x * 2)));
  ICHECK((px - floormod(py, px * PConst<PrimExpr>(2))).Match(x - floormod(2, x * 2)));

  // logicals
  ICHECK((px == pz).Match(x == 1));
  ICHECK((px != pz).Match(x != 1));
  ICHECK((px > py).Match(x > y));
  ICHECK((px < py).Match(x < y));
  ICHECK((px <= py).Match(x <= y));
  ICHECK((px >= py).Match(x >= y));
  ICHECK((px >= py && px < pz).Match(x >= y && x < z));
  ICHECK((!(px > py || px != py)).Match(!(x > y || x != y)));
  {
    ICHECK(select(px >= pz, py, py + pz).Match(tir::Select((x + 1) >= 1, y, y + 1)));
    ICHECK(tir::ExprDeepEqual()(px.Eval(), x + 1));
  }
  // bit intrinsics
  {
    ICHECK((px >> pz).Match(x >> 1));
    ICHECK(is_const_int(pz.Eval(), 1));
  }
  ICHECK(!(px >> pz).Match(x << 1));
  ICHECK((px << pz).Match(x << 1));
  ICHECK((px & pz).Match(x & 1));
  ICHECK((px | pz).Match(x | 1));
  ICHECK((px ^ pz).Match(x ^ 1));
  ICHECK((px - (~(py | (px * pz)))).Match(x - (~(2 | (x * 2)))));
  // select
  {
    ICHECK(select(px > pz, py, py + pz).Match(tir::Select(x > 1, y, y + 1)));
    ICHECK(is_const_int(pz.Eval(), 1));
  }
  ICHECK(!select(px > pz, py, py + pz).Match(tir::Select(x > 2, y, y + 1)));
  ICHECK(!select(px > pz, py, py).Match(tir::Select(x > 2, y, y + 1)));
  {
    ICHECK(select(px, py, pz).Match(tir::Select(x > 2, y, y + 1)));
    ICHECK(tir::ExprDeepEqual()(pz.Eval(), y + 1));
  }
  // if_then_else
  {
    ICHECK(if_then_else(px > pz, py, py + pz).Match(if_then_else(x > 1, y, y + 1)));
    ICHECK(is_const_int(pz.Eval(), 1));
  }
  // cast pattern
  {
    ICHECK(!cast(PConst<DataType>(DataType::Int(32)), px).Match(tir::Cast(DataType::Float(64), x)));
    ICHECK(cast(pt, px).Match(tir::Cast(DataType::Float(64), x)));
    ICHECK(pt.Eval() == DataType::Float(64));
    auto zz = cast(pt, px).Eval();
    ICHECK((cast(pt, px) - cast(pt, py))
               .Match(tir::Cast(DataType::Float(64), x) - tir::Cast(DataType::Int(64), x)));
    auto expr = tir::Cast(DataType::Int(32), tir::Cast(DataType::Float(64), x));
    ICHECK(!(cast(pt, cast(pt, px))).Match(expr));
  }
  // ramp pattern
  {
    ICHECK(ramp(px, PConst<PrimExpr>(1), planes).Match(tir::Ramp(x, 1, 10)));
    ICHECK(planes.Eval() == 10);
    ICHECK(!ramp(px, PConst<PrimExpr>(1), planes).Match(tir::Ramp(x, 2, 10)));
  }
  // broadcast pattern
  {
    ICHECK(broadcast(px, planes).Match(tir::Broadcast(x, 10)));
    ICHECK(planes.Eval() == 10);
    ICHECK(broadcast(px * py, planes).Match(tir::Broadcast(x * 10, 10)));
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
    ICHECK((v * c).Match(tx * 3));
    ICHECK_EQ(c.Eval()->value, 3);
    ICHECK((v * 3).Match(tx * 3));
  }
  // cannot match c to ty
  ICHECK(!(v * c).Match(tx * ty));
  // cannot match tx + 1 to v
  ICHECK(!(v * c).Match((tx + 1) * 3));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
