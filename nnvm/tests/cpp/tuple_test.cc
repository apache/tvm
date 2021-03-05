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
#include <nnvm/tuple.h>

TEST(Tuple, Basic) {
  using nnvm::TShape;
  using nnvm::Tuple;
  Tuple<int> x{1, 2, 3};
  Tuple<int> y{1, 2, 3, 5, 6};
  x = std::move(y);

  CHECK_EQ(x.ndim(), 5);
  Tuple<int> z{1, 2, 3, 5, 6};
  std::ostringstream os;
  os << z;
  CHECK_EQ(os.str(), "[1,2,3,5,6]");
  std::istringstream is(os.str());
  is >> y;
  CHECK_EQ(x, y);
  Tuple<nnvm::dim_t> ss{1, 2, 3};
  TShape s = ss;
  s = std::move(ss);
  CHECK((s == TShape{1, 2, 3}));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
