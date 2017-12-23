#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <arithmetic/Simplify.h>

TEST(IRSIMPLIFY, Basic) {
  using namespace HalideIR::Internal;
  simplify_test();
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
