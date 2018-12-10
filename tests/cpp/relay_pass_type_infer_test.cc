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
  auto f = relay::FunctionNode::make(tvm::Array<relay::Var>{ x }, x, type_b,
                                     Array<relay::TypeVar>{type_a, type_b});

  auto y = relay::VarNode::make("y", type_a);
  auto call = relay::CallNode::make(f, Array<relay::Expr>{ y });
  auto fx = relay::FunctionNode::make(tvm::Array<relay::Var>{ y }, call, type_b,
                                      Array<relay::TypeVar>{type_a, type_b});
  auto type_fx = relay::InferType(fx, relay::ModuleNode::make(Map<relay::GlobalVar, relay::Function>{}));

  auto expected = relay::FuncTypeNode::make(tvm::Array<relay::Type>{ type_a }, type_a,
                                            relay::Array<relay::TypeVar>{type_a} , {});
  CHECK_EQ(type_fx->checked_type(), expected);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
