#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Tensor, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A({m, l}, "A");
  Tensor B({n, l}, "B");

  auto C = Compute({m, n}, [&](Var i, Var j) {
      return A[i][j];
    }, "C");

  Tensor::Slice x = A[n];
  LOG(INFO) << C->op.as<ComputeOpNode>()->body;
}

TEST(Tensor, Reduce) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A({m, l}, "A");
  Tensor B({n, l}, "B");
  IterVar rv(Range{0, l}, "k");

  auto C = Compute({m, n}, [&](Var i, Var j) {
      return sum(max(1 + A[i][rv] + 1, B[j][rv]), {rv});
      }, "C");
  LOG(INFO) << C->op.as<ComputeOpNode>()->body;
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
