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

#include <gtest/gtest.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

TEST(MapRanges, CalcRanges) {
  using namespace tvm;
  using namespace tvm::tir;

  IndexMapNode node;
  arith::Analyzer local_analyzer;

  tvm::tir::Var i0("i0", DataType::Int(64));
  tvm::tir::Var i1("i1", DataType::Int(64));
  tvm::tir::Var i2("i2", DataType::Int(64));
  tvm::tir::Var i3("i3", DataType::Int(64));
  node.initial_indices.push_back(i0);
  node.initial_indices.push_back(i1);
  node.initial_indices.push_back(i2);
  node.initial_indices.push_back(i3);

  Array<Range> ranges;
  // The values were taken from the first layer of DeeplabV3.
  ranges.push_back(Range(PrimExpr(0), PrimExpr(1)));
  ranges.push_back(Range(PrimExpr(0), PrimExpr(1)));
  ranges.push_back(Range(PrimExpr(0), PrimExpr(144)));
  ranges.push_back(Range(PrimExpr(0), PrimExpr(48)));

  // the result of MetaSheduler trace
  node.final_indices.push_back(floordiv(i2, PrimExpr(12)));

  auto i2_48_i3 = add(mul(i2, PrimExpr(48)), i3);  // (i2 * 48 + i3)

  node.final_indices.push_back((floordiv(floormod(i2_48_i3, PrimExpr(64)), PrimExpr(32))));
  node.final_indices.push_back((floordiv(floormod(i2_48_i3, PrimExpr(3072)), PrimExpr(32))));

  auto i2_16_i3 = add(mul(i2, PrimExpr(16)), i3);  // (i2 * 16 + i3)
  node.final_indices.push_back(floormod(i2_16_i3, PrimExpr(32)));

  auto result = node.MapRanges(ranges, &local_analyzer);

  EXPECT_EQ(result.size(), ranges.size());

  PrimExpr orig_sz(1);
  PrimExpr total_sz(1);

  for (size_t i = 0; i < ranges.size(); ++i) {
    auto pRange = ranges[i].as<RangeNode>();
    EXPECT_NE(pRange, nullptr);
    orig_sz *= pRange->extent;

    auto pResult = result[i].as<RangeNode>();
    EXPECT_NE(pResult, nullptr);
    total_sz *= pResult->extent;
  }
  auto sz1 = local_analyzer.Simplify(total_sz);
  auto sz2 = local_analyzer.Simplify(orig_sz);
  auto pTotal = total_sz.as<IntImmNode>();
  auto pOrig = orig_sz.as<IntImmNode>();
  EXPECT_NE(pTotal, nullptr);
  EXPECT_NE(pOrig, nullptr);
  EXPECT_EQ(pOrig->value, pTotal->value);
}
