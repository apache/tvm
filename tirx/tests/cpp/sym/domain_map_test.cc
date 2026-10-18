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
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/sym/bound.h>
#include <tvm/sym/int_set.h>

namespace tvm {
namespace sym {
namespace {

void ExpectInterval(const IntSet& result, int min, int max) {
  Analyzer analyzer;
  EXPECT_TRUE(analyzer->CanProveEqual(result.min(), min));
  EXPECT_TRUE(analyzer->CanProveEqual(result.max(), max));
}

TEST(DomainMap, EvalSetOverloads) {
  PrimVar x("x");
  const ffi::Map<Var, IntSet> domains{{x, IntSet::Interval(2, 5)}};
  PrimExpr expr = x + 1;
  Range range = Range::FromMinExtent(x, 3);
  ExpectInterval(EvalSet(expr, domains), 3, 6);
  ExpectInterval(EvalSet(IntSet::Interval(x, x + 2), domains), 2, 7);
  ExpectInterval(EvalSet(range, domains), 2, 7);
  auto region = EvalSet(ffi::Array<Range>{range}, domains);
  ASSERT_EQ(region.size(), 1);
  ExpectInterval(region[0], 2, 7);
  auto subexpressions = EvalSetForEachSubExpr(expr, domains);
  ExpectInterval(subexpressions.at(expr), 3, 6);
  ExpectInterval(subexpressions.at(x), 2, 5);
  ExpectInterval(domains.at(x), 2, 5);
}

TEST(DomainMap, DeduceBoundWithRelaxation) {
  PrimVar x("x"), y("y");
  const ffi::Map<Var, IntSet> hints{{y, IntSet::Interval(2, 5)}};
  const ffi::Map<Var, IntSet> relaxed{{y, IntSet::Interval(2, 5)}};
  auto bound = DeduceBound(x, x + y < 10, hints, relaxed);
  Analyzer analyzer;
  EXPECT_TRUE(analyzer->CanProveEqual(bound.max(), 4));
  ExpectInterval(hints.at(y), 2, 5);
  ExpectInterval(relaxed.at(y), 2, 5);
}

}  // namespace
}  // namespace sym
}  // namespace tvm
