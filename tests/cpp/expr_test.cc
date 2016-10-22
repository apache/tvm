#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Expr, Basic) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  std::ostringstream os;
  os << z;
  CHECK(os.str() == "max(((x + 1) + 2), 100)");
}

TEST(Expr, Reduction) {
  using namespace tvm;
  Var x("x");
  RDomain rdom({{0, 3}});
  auto z = sum(x + 1 + 2, rdom);
  std::ostringstream os;
  os << z;
  CHECK(os.str() == "reduce(+, ((x + 1) + 2), rdomain([[0, 3)]))");
}

TEST(Expr, Simplify) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, x + 10) * 100;
  std::ostringstream os;
  os << Simplify(z);
  CHECK(os.str() == "((x * 100) + 1000)");
}

TEST(Expr, Bind) {
  using namespace tvm;
  Var x("x"), y("y"), z("z");
  Var i("i"), j("j");
  Tensor A({y, z}, "A");
  Expr e1 = x * 5;
  std::unordered_map<Expr, Expr> dict = {{x, y * 10 + z}};
  std::ostringstream os1, os2;
  os1 << Bind(e1, dict);
  CHECK(os1.str() == "(((y * 10) + z) * 5)");
  
  Expr e2 = A(i, j);
  dict.clear();
  dict[i] = 64 * x;
  dict[j] = z + 16 * y;
  os2 << Bind(e2, dict);
  CHECK(os2.str() == "A[(64 * x), (z + (16 * y))]");
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
