#include <tvm/tvm.h>
#include <topi/elemwise.h>
#include <gtest/gtest.h>

namespace topi {
TEST(Tensor, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A = placeholder({m, l}, Float(32), "A");
  auto C = topi::exp(A);
}
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
