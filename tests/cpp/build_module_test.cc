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
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/topi/cuda/injective.h>

#include <cmath>
#include <string>

TEST(BuildModule, Basic) {
  using namespace tvm;
  using namespace tvm::te;
  auto n = var("n");
  Array<PrimExpr> shape;
  shape.push_back(n);

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");

  auto C = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "C");

  auto s = create_schedule({C->op});

  auto cAxis = C->op.as<ComputeOpNode>()->axis;

  IterVar bx, tx;
  s[C].split(cAxis[0], 64, &bx, &tx);

  auto args = Array<Tensor>({A, B, C});
  std::unordered_map<Tensor, Buffer> binds;

  auto target = Target("llvm");

  auto lowered = LowerSchedule(s, args, "func", binds, GlobalVarSupply());
  auto module = build(lowered, target, Target());

  auto mali_target = Target("opencl -model=Mali-T860MP4@800Mhz -device=mali");
  ICHECK_EQ(mali_target->kind->name, "opencl");
  ICHECK_EQ(mali_target->keys.size(), 3);
  ICHECK_EQ(mali_target->keys[0], "mali");
  ICHECK_EQ(mali_target->keys[1], "opencl");
  ICHECK_EQ(mali_target->keys[2], "gpu");
  ICHECK_EQ(mali_target->GetAttr<String>("device").value(), "mali");
  ICHECK_EQ(mali_target->GetAttr<String>("model").value(), "Mali-T860MP4@800Mhz");
  ICHECK_EQ(mali_target->GetAttr<Integer>("max_num_threads").value(), 256);
}
