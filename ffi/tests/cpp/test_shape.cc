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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/shape.h>

namespace {

using namespace tvm::ffi;

TEST(Shape, Basic) {
  Shape shape = Shape({1, 2, 3});
  EXPECT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 2);
  EXPECT_EQ(shape[2], 3);

  Shape shape2 = Shape(Array<int64_t>({4, 5, 6, 7}));
  EXPECT_EQ(shape2.size(), 4);
  EXPECT_EQ(shape2[0], 4);
  EXPECT_EQ(shape2[1], 5);
  EXPECT_EQ(shape2[2], 6);
  EXPECT_EQ(shape2[3], 7);

  std::vector<int64_t> vec = {8, 9, 10};
  Shape shape3 = Shape(std::move(vec));
  EXPECT_EQ(shape3.size(), 3);
  EXPECT_EQ(shape3[0], 8);
  EXPECT_EQ(shape3[1], 9);
  EXPECT_EQ(shape3[2], 10);
  EXPECT_EQ(shape3.Product(), 8 * 9 * 10);

  Shape shape4 = Shape();
  EXPECT_EQ(shape4.size(), 0);
  EXPECT_EQ(shape4.Product(), 1);
}

TEST(Shape, AnyConvert) {
  Shape shape0 = Shape({1, 2, 3});
  Any any0 = shape0;

  auto shape1 = any0.cast<Shape>();
  EXPECT_EQ(shape1.size(), 3);
  EXPECT_EQ(shape1[0], 1);
  EXPECT_EQ(shape1[1], 2);
  EXPECT_EQ(shape1[2], 3);

  Array<Any> arr({1, 2});
  AnyView any_view0 = arr;
  auto shape2 = any_view0.cast<Shape>();
  EXPECT_EQ(shape2.size(), 2);
  EXPECT_EQ(shape2[0], 1);
  EXPECT_EQ(shape2[1], 2);
}

}  // namespace
