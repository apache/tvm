#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/ir_pass.h>

TEST(IRPass, Substitute) {
  using namespace Halide::Internal;
  using namespace tvm;
  Var x("x"), y;
  auto z = x + y;
  {
    auto zz = ir::Substitute({{y.get(), 11}}, z);
    std::ostringstream os;
    os << zz;
    CHECK(os.str() == "(x + 11)");
  }
  {
    auto zz = ir::Substitute({{z.get(), 11}}, z);
    std::ostringstream os;
    os << zz;
    CHECK(os.str() == "11");
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
