#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Array, Expr) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Array<Expr> list{x, z, z};
  LOG(INFO) << list.size();
  LOG(INFO) << list[0];
  LOG(INFO) << list[1];
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
