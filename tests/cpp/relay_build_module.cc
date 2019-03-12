#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/relay/pass.h>

TEST(Relay, BuildModile) {
  using namespace tvm;
  auto tensor_type = relay::TensorTypeNode::make({2, 3}, ::tvm::Float(32));
  auto a = relay::VarNode::make("a", tensor_type);
  auto b = relay::VarNode::make("b", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto x = relay::CallNode::make(add_op, {a, b}, tvm::Attrs(), {});
  auto c = relay::VarNode::make("c", tensor_type);
  auto y = relay::CallNode::make(add_op, {x, c}, tvm::Attrs(), {});
  auto func = relay::FunctionNode::make(relay::FreeVars(y), y, relay::Type(), {});
  auto A = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto B = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto C = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  // auto Y = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto pA = (float*)A.ToDLPack()->dl_tensor.data;
  auto pB = (float*)B.ToDLPack()->dl_tensor.data;
  auto pC = (float*)C.ToDLPack()->dl_tensor.data;
  // auto pY = (float*)Y.ToDLPack()->dl_tensor.data;
  for (int i = 0; i < 6; ++i) {
    pA[i] = i;
    pB[i] = i + 1;
    pC[i] = i + 2;
    // pY[i] = 0;
  }
  // build
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);
  build_f(func, "llvm", "llvm");
  std::string json = json_f();
  tvm::runtime::Module mod = mod_f();
  // run
  auto ctx = A->ctx;
  auto pfr = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
  tvm::runtime::Module run_mod = (*pfr)(json, mod, (int)ctx.device_type, (int)ctx.device_id);
  auto set_input_f = run_mod.GetFunction("set_input", false);
  auto run_f = run_mod.GetFunction("run", false);
  auto get_output_f = run_mod.GetFunction("get_output", false);
  set_input_f("a", A);
  set_input_f("b", B);
  set_input_f("c", C);
  run_f();
  tvm::runtime::NDArray Y = get_output_f(0);
  auto pY = (float*)Y.ToDLPack()->dl_tensor.data;
  for (int i = 0; i < 6; ++i) {
    CHECK_EQ(pY[i], i + (i + 1) + (i + 2));
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
