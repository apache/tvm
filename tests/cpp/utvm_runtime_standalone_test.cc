/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <random>

#include <dlpack/dlpack.h>
#include <gtest/gtest.h>
#include <map>
#include <vector>

#ifdef USE_MICRO_STANDALONE_RUNTIME

// Use system(..), `gcc -shared -fPIC`, thus restrict the test to OS X for now.
#if defined(__APPLE__) && defined(__MACH__)

#include <gtest/gtest.h>
#include <topi/generic/injective.h>
#include <tvm/driver/driver_api.h>
#include <tvm/te/operation.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/micro/standalone/utvm_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <spawn.h>
#include <sys/wait.h>

TVM_REGISTER_GLOBAL("test.sch").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  *rv = topi::generic::schedule_injective(args[0], args[1]);
});

TEST(MicroStandaloneRuntime, BuildModule) {
  using namespace tvm;
  auto tensor_type = relay::TensorType({2, 3}, ::tvm::Float(32));
  auto a = relay::VarNode::make("a", tensor_type);
  auto b = relay::VarNode::make("b", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto x = relay::CallNode::make(add_op, {a, b}, tvm::Attrs(), {});
  auto c = relay::VarNode::make("c", tensor_type);
  auto y = relay::CallNode::make(add_op, {x, c}, tvm::Attrs(), {});
  auto func = relay::Function(relay::FreeVars(y), y, relay::Type(), {});
  auto A = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto B = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto C = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  auto pA = (float*)A->data;
  auto pB = (float*)B->data;
  auto pC = (float*)C->data;

  for (int i = 0; i < 6; ++i) {
    pA[i] = i;
    pB[i] = i + 1;
    pC[i] = i + 2;
  }
  // get schedule
  auto reg = tvm::runtime::Registry::Get("relay.op._Register");
  auto s_i = tvm::runtime::Registry::Get("test.sch");
  if (!reg) {
    LOG(FATAL) << "no _Register";
  }
  if (!s_i) {
    LOG(FATAL) << "no test_sch";
  }
  (*reg)("add", "FTVMSchedule", *s_i, 10);
  // build
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);
  Map<tvm::Integer, tvm::Target> targets;

  Target llvm_tgt = Target::Create("llvm");
  targets.Set(0, llvm_tgt);
  build_f(func, targets, llvm_tgt);
  std::string json = json_f();
  tvm::runtime::Module mod = mod_f();
  std::string o_fname = std::tmpnam(nullptr);
  std::string so_fname = std::tmpnam(nullptr);
  mod->SaveToFile(o_fname, "o");
  const std::vector<std::string> args = {"gcc", "-shared", "-fPIC", "-o", so_fname, o_fname};
  std::stringstream s;
  for (auto& c : args) {
    s << c << " ";
  }
  const auto ss = s.str();
  const auto ret = system(ss.c_str());
  ASSERT_EQ(ret, 0);
  // Now, execute the minimal runtime.
  auto* dsoModule = UTVMRuntimeDSOModuleCreate(so_fname.c_str(), so_fname.size());
  ASSERT_NE(dsoModule, nullptr);
  auto* handle = UTVMRuntimeCreate(json.c_str(), json.size(), dsoModule);
  ASSERT_NE(handle, nullptr);

  UTVMRuntimeSetInput(handle, 0, &A.ToDLPack()->dl_tensor);
  UTVMRuntimeSetInput(handle, 1, &B.ToDLPack()->dl_tensor);
  UTVMRuntimeSetInput(handle, 2, &C.ToDLPack()->dl_tensor);
  UTVMRuntimeRun(handle);
  auto Y = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  UTVMRuntimeGetOutput(handle, 0, &Y.ToDLPack()->dl_tensor);
  auto* pY = (float*)Y->data;
  for (int i = 0; i < 6; ++i) {
    CHECK_LT(fabs(pY[i] - (i + (i + 1) + (i + 2))), 1e-4);
  }
  UTVMRuntimeDestroy(handle);
  UTVMRuntimeDSOModuleDestroy(dsoModule);
}

#endif
#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
