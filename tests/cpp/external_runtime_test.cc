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

/*!
 * \file external_runtime_test.cc
 * \brief Test an example runtime module to interpreting a json string.
 */
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/contrib/gcc.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cmath>
#include <sstream>
#include <string>

using tvm::runtime::ExampleJSonModule;
using tvm::runtime::Module;
using tvm::runtime::ModuleNode;
using tvm::runtime::NDArray;
using tvm::runtime::Object;
using tvm::runtime::ObjectPtr;
using tvm::runtime::PackedFunc;
using tvm::runtime::TVMArgs;
using tvm::runtime::TVMArgsSetter;
using tvm::runtime::TVMRetValue;

TEST(ExampleModule, Basic) {
  // This is a simple json format used for testing. Users/vendors can define
  // their own format.
  std::string json =
      "gcc_0\n"
      "input 0 10 10\n"
      "input 1 10 10\n"
      "input 2 10 10\n"
      "add 3 inputs: 0 1 shape: 10 10\n"
      "sub 4 inputs: 3 2 shape: 10 10";

  Module mod = ExampleJSonModule::LoadFromFile(json, "");
  PackedFunc f = mod.GetFunction("gcc_0", false);

  auto a_val = NDArray::Empty({10, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto b_val = NDArray::Empty({10, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto c_val = NDArray::Empty({10, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  float* pa = (float*)a_val.ToDLPack()->dl_tensor.data;
  float* pb = (float*)b_val.ToDLPack()->dl_tensor.data;
  float* pc = (float*)c_val.ToDLPack()->dl_tensor.data;

  // Assign values.
  for (int i = 0; i < 10 * 10; i++) {
    pa[i] = i;
    pb[i] = i + 1.0;
    pc[i] = i + 2.0;
  }

  NDArray out = f(a_val, b_val, c_val);
  float* p_out = (float*)out.ToDLPack()->dl_tensor.data;

  // Check correctness of result
  for (int i = 0; i < 10; i++) {
    CHECK_LT(std::fabs(p_out[i] - ((i + (i + 1.0) - (i + 2.0)))), 1e-5);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
