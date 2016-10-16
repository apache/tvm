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

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
