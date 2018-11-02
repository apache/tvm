#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/relay/pass.h>

TEST(Relay, SelfReference) {
  using namespace tvm;
  auto type_a = relay::TypeVarNode::make("a", relay::TypeVarNode::kType);
  auto type_b = relay::TypeVarNode::make("b", relay::TypeVarNode::kType);
  auto x = relay::VarNode::make("x", type_a);
  auto f = relay::FunctionNode::make(tvm::Array<relay::Var>{ x }, x, type_b, Array<relay::TypeVar>{});
  auto fx = relay::CallNode::make(f, Array<relay::Expr>{ x });
  auto type_fx = relay::InferType(fx, relay::ModuleNode::make(Map<relay::GlobalVar, relay::Function>{}));
  CHECK_EQ(type_fx->checked_type(), type_a);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
