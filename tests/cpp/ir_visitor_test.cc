#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

TEST(IRVisitor, CountVar) {
  using namespace HalideIR::Internal;
  using namespace tvm;
  int n_var = 0;
  Var x("x"), y;

  auto z = x + 1 + y + y;
  ir::PostOrderVisit(z, [&n_var](const NodeRef& n) {
      if (n.as<Variable>()) ++n_var;
    });
  CHECK_EQ(n_var, 2);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
