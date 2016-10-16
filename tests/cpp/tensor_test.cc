#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Tensor, Basic) {
  using namespace tvm;
  Var m, n, k;
  Tensor A({m, k});
  Tensor B({n, k});

  auto x = [=](Var i, Var j, Var k) {
    return A(i, k) * B(j, k);
  };
  auto C = Tensor({m, n}, x);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
