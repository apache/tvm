#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/build_module.h>

TEST(BuildModule, Basic) {
  using namespace tvm;
  auto n = var("n");
  Array<Expr> shape;
  shape.push_back(n);

  auto A = placeholder(shape, Float(32), "A");
  auto B = placeholder(shape, Float(32), "B");

  auto C = compute(A->shape, [&A, &B](Expr i) {
    return A[i] + B[i];
  }, "C");

  auto s = create_schedule({ C->op });

  auto cAxis = C->op.as<ComputeOpNode>()->axis;

  IterVar bx, tx;
  s[C].split(cAxis[0], 64, &bx, &tx);

  auto args = Array<Tensor>({ A, B, C });
  std::unordered_map<Tensor, Buffer> binds;

  auto config = build_config();
  auto target = target::llvm();

  auto lowered = lower(s, args, "func", binds, config);
  auto module = build(lowered, target, Target(), config);

  auto mali_target = Target::create("opencl -model=Mali-T860MP4@800Mhz -device=mali");
  CHECK_EQ(mali_target->str(), "opencl -model=Mali-T860MP4@800Mhz -device=mali"); 
}


int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
