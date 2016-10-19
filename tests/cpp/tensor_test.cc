
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Tensor, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A({m, l}, "A");
  Tensor B({n, l}, "B");
  RDomain rd({{0, l}});

  auto C = Tensor({m, n}, [&](Var i, Var j) {
      return sum(A(i, rd.i0()) * B(j, rd.i0()), rd);
    }, "C");

  auto inputs = C.InputTensors();
  CHECK(inputs[0] == A);
  CHECK(inputs[1] == B);
  CHECK(C.IsRTensor());
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
