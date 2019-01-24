#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/ir_pass.h>
#include <tvm/tvm.h>
#include <arithmetic/Simplify.h>

TEST(IRSIMPLIFY, Basic) {
  using namespace HalideIR::Internal;
  simplify_test();
}

TEST(IRSIMPLIFY, MinMax) {
  auto x = tvm::var("x");
  auto e1 = (tvm::max(x, 1) - tvm::max(x, 1)) ;
  auto e1s = tvm::ir::CanonicalSimplify(e1);
  CHECK(is_zero(e1s));

  auto e2 = (x * tvm::min(x, 1)) - (x * tvm::min(x, 1));
  auto e2s = tvm::ir::CanonicalSimplify(e2);
  CHECK(is_zero(e2s));
}

TEST(IRSIMPLIFY, Mul) {
  auto x = tvm::var("x");
  auto e = (x * x) - (x * x) ;
  auto es = tvm::ir::CanonicalSimplify(e);
  CHECK(is_zero(es));
}

TEST(IRSIMPLIFY, Mod) {
  auto x = tvm::Integer(10);
  auto y = tvm::Integer(12);
  // Mod::make is used instead of % to avoid constant folding during
  // calling operator%(x,y). Mod::make doesn't try constant folding,
  // and therefore, the constant folding will be attempted in CanonicalSimplify
  auto mod = tvm::ir::CanonicalSimplify(tvm::ir::Mod::make(x, y));
  auto es = tvm::ir::CanonicalSimplify(mod - x);
  CHECK(is_zero(es));
}
int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
