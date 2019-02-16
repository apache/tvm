#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/relay/pass.h>

TEST(Relay, SelfReference) {
  using namespace tvm;
  auto tensor_type = relay::TensorTypeNode::make({}, ::tvm::Bool());
  auto x = relay::VarNode::make("x", relay::Type());
  auto f = relay::FunctionNode::make(tvm::Array<relay::Var>{ x }, x, relay::Type(), {});

  auto y = relay::VarNode::make("y", tensor_type);
  auto call = relay::CallNode::make(f, Array<relay::Expr>{ y });
  auto fx = relay::FunctionNode::make(tvm::Array<relay::Var>{ y }, call, relay::Type(), {});
  auto empty_module =
    relay::ModuleNode::make(Map<relay::GlobalVar, relay::Function>{},
                            Map<relay::GlobalTypeVar, relay::TypeData>{});
  auto type_fx = relay::InferType(fx, empty_module);

  auto expected = relay::FuncTypeNode::make(tvm::Array<relay::Type>{ tensor_type }, tensor_type, {}, {});
  CHECK(AlphaEqual(type_fx->checked_type(), expected));
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
