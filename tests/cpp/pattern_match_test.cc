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

#include <gtest/gtest.h>
#include "../src/arith/pattern_match.h"

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
  CHECK(!(px + (px + px)).Match(r));
  CHECK(!(px + (py + py)).Match(r));
  CHECK((px + (py + pz)).Match(r));
  auto pattern = px + (py + pz);
  CHECK(pattern.Match(r));
  {
    CHECK((px + (py + px)).Match(r));
    auto rr = (px + py).Eval();
    CHECK(tir::Equal(rr, 1 + y));
    CHECK(tir::Equal(px.Eval() + py.Eval(), 1 + y));
  }
  {
    CHECK((px + max(py, px)).Match((x + 1) + max(y, (x + 1))));
    CHECK(tir::Equal(px.Eval(), x + 1));
  }
  CHECK(!(px + min(py, px)).Match((x + 1) + max(y, (x + 1))));
  CHECK((px + min(py, px)).Match(z + min(y, z)));
  CHECK((px + truncdiv(py, px * py)).Match(x + truncdiv(2, x * 2)));
  CHECK((px - truncmod(py, px * pz)).Match(x - truncmod(2, x * 2)));
  CHECK((px - floormod(py, px * PConst<PrimExpr>(2))).Match(x - floormod(2, x * 2)));

  // logicals
  CHECK((px == pz).Match(x == 1));
  CHECK((px != pz).Match(x != 1));
  CHECK((px > py).Match(x > y));
  CHECK((px < py).Match(x < y));
  CHECK((px <= py).Match(x <= y));
  CHECK((px >= py).Match(x >= y));
  CHECK((px >= py && px < pz).Match(x >= y && x < z));
  CHECK((!(px > py || px != py)).Match(!(x > y || x != y)));
  {
    CHECK(select(px >= pz, py, py + pz).Match(
        tir::SelectNode::make((x + 1) >= 1, y, y + 1)));
    CHECK(tir::Equal(px.Eval(), x + 1));
  }
  // bit intrinsics
  {
    CHECK((px >> pz).Match(x >> 1));
    CHECK(is_const_int(pz.Eval(), 1));
  }
  CHECK(!(px >> pz).Match(x << 1));
  CHECK((px << pz).Match(x << 1));
  CHECK((px & pz).Match(x & 1));
  CHECK((px | pz).Match(x | 1));
  CHECK((px ^ pz).Match(x ^ 1));
  CHECK((px - (~(py | (px * pz)))).Match(x - (~(2 | (x * 2)))));
  // select
  {
    CHECK(select(px > pz, py, py + pz).Match(
      tir::SelectNode::make(x > 1, y, y + 1)));
    CHECK(is_const_int(pz.Eval(), 1));
  }
  CHECK(!select(px > pz, py, py + pz).Match(
      tir::SelectNode::make(x > 2, y, y + 1)));
  CHECK(!select(px > pz, py, py).Match(
      tir::SelectNode::make(x > 2, y, y + 1)));
  {
    CHECK(select(px, py, pz).Match(
        tir::SelectNode::make(x > 2, y, y + 1)));
    CHECK(tir::Equal(pz.Eval(), y + 1));
  }
  // if_then_else
  {
    CHECK(if_then_else(px > pz, py, py + pz).Match(
        if_then_else(x > 1, y, y + 1)));
    CHECK(is_const_int(pz.Eval(), 1));
  }
  // cast pattern
  {
    CHECK(!cast(PConst<DataType>(
        DataType::Int(32)), px).Match(tir::CastNode::make(DataType::Float(64), x)));
    CHECK(cast(pt, px).Match(tir::CastNode::make(DataType::Float(64), x)));
    CHECK(pt.Eval() == DataType::Float(64));
    auto zz = cast(pt, px).Eval();
    CHECK((cast(pt, px) - cast(pt, py)).Match(
        tir::CastNode::make(DataType::Float(64), x) - tir::CastNode::make(DataType::Int(64), x)));
    auto expr = tir::CastNode::make(DataType::Int(32), tir::CastNode::make(DataType::Float(64), x));
    CHECK(!(cast(pt, cast(pt, px))).Match(expr));
  }
  // ramp pattern
  {
    CHECK(ramp(px, PConst<PrimExpr>(1), planes).Match(
        tir::RampNode::make(x, 1, 10)));
    CHECK(planes.Eval() == 10);
    CHECK(!ramp(px, PConst<PrimExpr>(1), planes).Match(
        tir::RampNode::make(x, 2, 10)));
  }
  // broadcast pattern
  {
    CHECK(broadcast(px, planes).Match(
        tir::BroadcastNode::make(x, 10)));
    CHECK(planes.Eval() == 10);
    CHECK(broadcast(px * py , planes).Match(
        tir::BroadcastNode::make(x * 10, 10)));
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
    CHECK((v * c).Match(tx * 3));
    CHECK_EQ(c.Eval()->value, 3);
    CHECK((v * 3).Match(tx * 3));
  }
  // cannot match c to ty
  CHECK(!(v * c).Match(tx * ty));
  // cannot match tx + 1 to v
  CHECK(!(v * c).Match((tx + 1) * 3));
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
