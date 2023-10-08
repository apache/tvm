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

#if TVM_MLIR_VERSION >= 150
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/arith/analyzer.h>
#include <tvm/te/operation.h>

#include "../src/arith/presburger_set.h"

TEST(PresburgerSet, eval) {
  auto x = tvm::tir::Var("x");
  auto y = tvm::tir::Var("y");
  auto sub_constraint0 = (x + y < 20) && (x - y <= 0);
  auto sub_constraint1 = x >= 0 && x < 20 && y >= 0 && y < 20;
  auto constraint = sub_constraint0 && sub_constraint1;
  auto set = tvm::arith::PresburgerSet(constraint);

  auto target = x + 2 * y;
  auto result = EvalSet(target, set);
  ASSERT_TRUE(tvm::tir::is_zero(result.min()));
  ASSERT_TRUE(tvm::tir::is_const_int(result.max(), 38));
}
#endif  // TVM_MLIR_VERSION
