#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/build_module.h>

TEST(BuildConfig, Serialization) {
  using namespace tvm;
  tvm::BuildConfig a;
  a.detect_global_barrier = true;
  a.offset_factor = 20;
  a.auto_unroll_max_depth = 2;
  auto s = a.str();
  auto b = tvm::BuildConfig::create(s);
  CHECK_EQ(a.data_alignment, b.data_alignment);
  CHECK_EQ(a.offset_factor, b.offset_factor);
  CHECK_EQ(a.double_buffer_split_loop, b.double_buffer_split_loop);
  CHECK_EQ(a.auto_unroll_max_step, b.auto_unroll_max_step);
  CHECK_EQ(a.auto_unroll_max_depth, b.auto_unroll_max_depth);
  CHECK_EQ(a.auto_unroll_max_extent, b.auto_unroll_max_extent);
  CHECK_EQ(a.unroll_explicit, b.unroll_explicit);
  CHECK_EQ(a.restricted_func, b.restricted_func);
  CHECK_EQ(a.detect_global_barrier, b.detect_global_barrier);
  CHECK_EQ(a.partition_const_loop, b.partition_const_loop);
}

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

  BuildConfig config;
  auto target = target::llvm();

  auto lowered = lower(s, args, "func", binds, config);
  auto module = build(lowered, target, nullptr, config);
}


int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
