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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/build_module.h>

TEST(BuildModule, Basic) {
  using namespace tvm;
  auto n = var("n");
  Array<Expr> shape;
  shape.push_back(n);

  auto A = placeholder(shape, Float(32), "A");
  auto B = placeholder(shape, Float(32), "B");

  auto C = compute(A->shape, [&A, &B](Expr i) {
    return A[i] + B[i];
  }, "C");

  auto s = create_schedule({ C->op });

  auto cAxis = C->op.as<ComputeOpNode>()->axis;

  IterVar bx, tx;
  s[C].split(cAxis[0], 64, &bx, &tx);

  auto args = Array<Tensor>({ A, B, C });
  std::unordered_map<Tensor, Buffer> binds;

  auto config = build_config();
  auto target = target::llvm();

  auto lowered = lower(s, args, "func", binds, config);
  auto module = build(lowered, target, Target(), config);

  auto mali_target = Target::create("opencl -model=Mali-T860MP4@800Mhz -device=mali");
  CHECK_EQ(mali_target->str(), "opencl -model=Mali-T860MP4@800Mhz -device=mali"); 
}


int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
