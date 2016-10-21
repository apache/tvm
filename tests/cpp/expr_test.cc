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

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
