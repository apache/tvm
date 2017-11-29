#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/build_module.h>

TEST(BuildModule, Basic) {
  using namespace tvm;
  auto n = Variable::make(Int(32), "n");
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

  s[C].bind(bx, thread_axis(Range(), "blockIdx.x"));
  s[C].bind(tx, thread_axis(Range(), "threadIdx.x"));


  auto args = Array<Tensor>({ A, B, C });
  std::unordered_map<Tensor, Buffer> binds;

  BuildConfig config;
  auto target = target::cuda();

  auto lowered = lower(s, args, "func", binds, config);
  auto module = build(lowered, target, nullptr, config);
}


int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
