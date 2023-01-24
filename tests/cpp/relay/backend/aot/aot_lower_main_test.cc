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

#include "../../../../../src/relay/backend/aot/aot_lower_main.h"

#include <gtest/gtest.h>
#include <tvm/relay/parser.h>

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

TEST(AOTLowerMain, ExprAllocatorSkipNestedFunc) {
  constexpr const char* mod_text = R"(
      #[version = "0.0.5"]
      def @main(%x: Tensor[(10, 10), float32]) {
        %0 = fn (%FunctionVar_01: Tensor[(10, 10), float32]) {
          nn.relu(%FunctionVar_01)
        };
        %0(%x)
      }
    )";
  IRModule mod = ParseModule("string", mod_text, {}, {});
  auto host_target = tvm::Target("llvm");
  auto prim_target = tvm::Target(host_target, host_target);
  auto ctxt = tvm::transform::PassContext::Current();
  auto config = tvm::CompilationConfig(ctxt, {prim_target});
  mod = tvm::relay::transform::PlanDevices(config)(mod);
  mod = tvm::relay::transform::InferType()(mod);

  StorageMap storage_map;
  std::vector<int> return_sids;
  auto func = Downcast<Function>(mod->Lookup("main"));
  std::tie(storage_map, return_sids) = CreateStorage(func);

  auto nested_func = Downcast<Function>(Downcast<Call>(func->body)->op);
  EXPECT_EQ(storage_map.find(nested_func->body), storage_map.end());
  EXPECT_EQ(storage_map.find(nested_func->params[0]), storage_map.end());
  EXPECT_NE(storage_map.find(func->body), storage_map.end());
  EXPECT_NE(storage_map.find(func->params[0]), storage_map.end());
}

}  // namespace aot
}  // namespace backend
}  // namespace relay
}  // namespace tvm
