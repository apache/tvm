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
#include <tvm/arith/analyzer.h>
#include <tvm/te/operation.h>

TEST(Simplify, MinMax) {
  tvm::arith::Analyzer ana;
  auto x = tvm::te::var("x");
  auto e1 = (tvm::max(x, 1) - tvm::max(x, 1));
  auto e1s = ana.canonical_simplify(e1);
  ICHECK(tvm::tir::is_zero(e1s));

  auto e2 = (x * tvm::min(x, 1)) - (x * tvm::min(x, 1));
  auto e2s = ana.canonical_simplify(e2);
  ICHECK(tvm::tir::is_zero(e2s));
}

TEST(Simplify, Mul) {
  tvm::arith::Analyzer ana;
  auto x = tvm::te::var("x");
  auto e = (x * x) - (x * x);
  auto es = ana.canonical_simplify(e);
  ICHECK(tvm::tir::is_zero(es));
}

TEST(Simplify, Mod) {
  tvm::arith::Analyzer ana;
  auto x = tvm::Integer(10);
  auto y = tvm::Integer(12);
  // Mod::make is used instead of % to avoid constant folding during
  // calling operator%(x,y). Mod::make doesn't try constant folding,
  // and therefore, the constant folding will be attempted in CanonicalSimplify
  auto mod = ana.canonical_simplify(tvm::tir::Mod(x, y));
  auto es = ana.canonical_simplify(mod - x);
  ICHECK(tvm::tir::is_zero(es));
}

TEST(ConstantFold, Broadcast) {
  tvm::StructuralEqual checker;
  auto i32x4 = tvm::tir::Broadcast(tvm::IntImm(tvm::DataType::Int(32), 10), 4);
  auto i64x4 = tvm::cast(i32x4->dtype.with_bits(64), i32x4);
  auto i64x4_expected = tvm::tir::Broadcast(tvm::IntImm(tvm::DataType::Int(64), 10), 4);
  ASSERT_TRUE(checker(i64x4, i64x4_expected));
}

TEST(ConstantFold, Ramp) {
  tvm::StructuralEqual checker;
  auto i32x4 = tvm::tir::Ramp(tvm::IntImm(tvm::DataType::Int(32), 10),
                              tvm::IntImm(tvm::DataType::Int(32), 1), 4);
  auto i64x4 = tvm::cast(i32x4->dtype.with_bits(64), i32x4);
  auto i64x4_expected = tvm::tir::Ramp(tvm::IntImm(tvm::DataType::Int(64), 10),
                                       tvm::IntImm(tvm::DataType::Int(64), 1), 4);
  ASSERT_TRUE(checker(i64x4, i64x4_expected));

  auto f32x4 = tvm::cast(tvm::DataType::Float(32, 4), i32x4);
  auto f32x4_expected = tvm::tir::Cast(tvm::DataType::Float(32, 4), i32x4);
  ASSERT_TRUE(checker(f32x4, f32x4_expected));
}
