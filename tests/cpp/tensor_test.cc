
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Tensor, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A({m, l}, "A");
  Tensor B({n, l}, "B");

  auto C = Tensor({m, n}, [&](Var i, Var j) {
      return A(i, j) * B(j, i);
    }, "C");
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
