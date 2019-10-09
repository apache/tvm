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
#include <tvm/operation.h>

TEST(Tensor, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");

  Tensor A = placeholder({m, l}, Float(32), "A");
  Tensor B = placeholder({n, l}, Float(32), "B");

  auto C = compute({m, n}, [&](Var i, Var j) {
      return A[i][j];
    }, "C");

  Tensor::Slice x = A[n];
}

TEST(Tensor, Reduce) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");
  Tensor A = placeholder({m, l}, Float(32), "A");
  Tensor B = placeholder({n, l}, Float(32), "B");
  IterVar rv = reduce_axis(Range{0, l}, "k");

  auto C = compute({m, n}, [&](Var i, Var j) {
      return sum(max(1 + A[i][rv] + 1, B[j][rv]), {rv});
      }, "C");
  LOG(INFO) << C->op.as<ComputeOpNode>()->body;
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
