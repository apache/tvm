#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Tensor, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");

  Tensor A = placeholder({m, l}, Float(32), "A");
  Tensor B = placeholder({n, l}, Float(32), "B");

  auto C = compute({m, n}, [&](Var i, Var j) {
      return A[i][j];
    }, "C");

  Tensor::Slice x = A[n];
}

TEST(Tensor, Reduce) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A = placeholder({m, l}, Float(32), "A");
  Tensor B = placeholder({n, l}, Float(32), "B");
  IterVar rv = reduce_axis(Range{0, l}, "k");

  auto C = compute({m, n}, [&](Var i, Var j) {
      return sum(max(1 + A[i][rv] + 1, B[j][rv]), {rv});
      }, "C");
  LOG(INFO) << C->op.as<ComputeOpNode>()->body;
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
