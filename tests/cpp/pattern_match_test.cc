#include <gtest/gtest.h>
#include "../src/arithmetic/pattern_match.h"

TEST(Pattern, Basic) {
  using namespace tvm;
  using namespace tvm::arith;
  Var x("x"), y("y"), z("z");
  arith::PVar<Expr> px, py, pz;
  arith::PVar<Type> pt;
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
    CHECK(ir::Equal(rr, 1 + y));
    CHECK(ir::Equal(px.Eval() + py.Eval(), 1 + y));
  }
  {
    CHECK((px + max(py, px)).Match((x + 1) + max(y, (x + 1))));
    CHECK(ir::Equal(px.Eval(), x + 1));
  }
  CHECK(!(px + min(py, px)).Match((x + 1) + max(y, (x + 1))));
  CHECK((px + min(py, px)).Match(z + min(y, z)));
  CHECK((px + py / (px * py)).Match(x + 2 / (x * 2)));
  CHECK((px - py % (px * pz)).Match(x - 2 % (x * 2)));
  CHECK((px - py % (px * PConst<Expr>(2))).Match(x - 2 % (x * 2)));

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
        ir::Select::make((x + 1) >= 1, y, y + 1)));
    CHECK(ir::Equal(px.Eval(), x + 1));
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
      ir::Select::make(x > 1, y, y + 1)));
    CHECK(is_const_int(pz.Eval(), 1));
  }
  CHECK(!select(px > pz, py, py + pz).Match(
      ir::Select::make(x > 2, y, y + 1)));
  CHECK(!select(px > pz, py, py).Match(
      ir::Select::make(x > 2, y, y + 1)));
  {
    CHECK(select(px, py, pz).Match(
        ir::Select::make(x > 2, y, y + 1)));
    CHECK(ir::Equal(pz.Eval(), y + 1));
  }
  // if_then_else
  {
    CHECK(if_then_else(px > pz, py, py + pz).Match(
        if_then_else(x > 1, y, y + 1)));
    CHECK(is_const_int(pz.Eval(), 1));
  }
  // cast pattern
  {
    CHECK(!cast(PConst<Type>(Int(32)), px).Match(ir::Cast::make(Float(64), x)));
    CHECK(cast(pt, px).Match(ir::Cast::make(Float(64), x)));
    CHECK(pt.Eval() == Float(64));
    auto zz = cast(pt, px).Eval();
    CHECK((cast(pt, px) - cast(pt, py)).Match(
        ir::Cast::make(Float(64), x) - ir::Cast::make(Int(64), x)));
    auto expr = ir::Cast::make(Int(32), ir::Cast::make(Float(64), x));
    CHECK(!(cast(pt, cast(pt, px))).Match(expr));
  }
  // ramp pattern
  {
    CHECK(ramp(px, PConst<Expr>(1), planes).Match(
        ir::Ramp::make(x, 1, 10)));
    CHECK(planes.Eval() == 10);
    CHECK(!ramp(px, PConst<Expr>(1), planes).Match(
        ir::Ramp::make(x, 2, 10)));
  }
  // broadcast pattern
  {
    CHECK(broadcast(px, planes).Match(
        ir::Broadcast::make(x, 10)));
    CHECK(planes.Eval() == 10);
    CHECK(broadcast(px * py , planes).Match(
        ir::Broadcast::make(x * 10, 10)));
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
