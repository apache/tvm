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
#include <tvm/tir/ir_pass.h>
#include <tvm/te/operation.h>

TEST(IRSIMPLIFY, MinMax) {
  auto x = tvm::te::var("x");
  auto e1 = (tvm::max(x, 1) - tvm::max(x, 1)) ;
  auto e1s = tvm::tir::CanonicalSimplify(e1);
  CHECK(tvm::tir::is_zero(e1s));

  auto e2 = (x * tvm::min(x, 1)) - (x * tvm::min(x, 1));
  auto e2s = tvm::tir::CanonicalSimplify(e2);
  CHECK(tvm::tir::is_zero(e2s));
}

TEST(IRSIMPLIFY, Mul) {
  auto x = tvm::te::var("x");
  auto e = (x * x) - (x * x) ;
  auto es = tvm::tir::CanonicalSimplify(e);
  CHECK(tvm::tir::is_zero(es));
}

TEST(IRSIMPLIFY, Mod) {
  auto x = tvm::Integer(10);
  auto y = tvm::Integer(12);
  // Mod::make is used instead of % to avoid constant folding during
  // calling operator%(x,y). Mod::make doesn't try constant folding,
  // and therefore, the constant folding will be attempted in CanonicalSimplify
  auto mod = tvm::tir::CanonicalSimplify(tvm::tir::ModNode::make(x, y));
  auto es = tvm::tir::CanonicalSimplify(mod - x);
  CHECK(tvm::tir::is_zero(es));
}
int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
