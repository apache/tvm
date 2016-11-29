#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/ir_functor.h>

TEST(IRF, Basic) {
  using namespace Halide::Internal;
  using namespace tvm;
  Var x("x");
  auto z = x + 1;

  IRFunctor<int(const NodeRef& n, int b)> f;
  LOG(INFO) << "x";
  f.set_dispatch<Variable>([](const Variable* n, int b) {
      return b;
    });
  f.set_dispatch<Add>([](const Add* n, int b) {
      return b + 2;
    });
  CHECK_EQ(f(x, 2),  2);
  CHECK_EQ(f(z, 2),  4);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
