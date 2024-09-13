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
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/optional.h>
#include <tvm/ffi/memory.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Optional, TInt) {
  Optional<TInt> x;
  Optional<TInt> y = TInt(11);
  static_assert(sizeof(Optional<TInt>) == sizeof(ObjectRef));

  EXPECT_TRUE(!x.has_value());
  EXPECT_TRUE(x == nullptr);
  EXPECT_EQ(x.value_or(TInt(12))->value, 12);

  EXPECT_TRUE(y.has_value());
  EXPECT_TRUE(y != nullptr);
  EXPECT_EQ(y.value_or(TInt(12))->value, 11);
}

TEST(Optional, double) {
  Optional<double> x;
  Optional<double> y = 11.0;
  static_assert(sizeof(Optional<double>) > sizeof(ObjectRef));

  EXPECT_TRUE(!x.has_value());
  EXPECT_TRUE(x == nullptr);
  EXPECT_EQ(x.value_or(12), 12);
  EXPECT_TRUE(x != 12);

  EXPECT_TRUE(y.has_value());
  EXPECT_TRUE(y != nullptr);
  EXPECT_EQ(y.value_or(12), 11);
  EXPECT_TRUE(y == 11);
  EXPECT_TRUE(y != 12);
}

TEST(Optional, AnyConvert_int) {
  Optional<int> opt_v0 = 1;
  EXPECT_EQ(opt_v0.value(), 1);
  EXPECT_TRUE(opt_v0 != nullptr);

  AnyView view0 = opt_v0;
  EXPECT_EQ(view0.operator int(), 1);

  Any any1;
  Optional<int> opt_v1 = any1;

  EXPECT_TRUE(opt_v1 == nullptr);
}

TEST(Optional, AnyConvert_Array) {
  AnyView view0;
  Array<Array<TNumber>> arr_nested = {{}, {TInt(1), TFloat(2)}};
  view0 = arr_nested;

  Optional<Array<Array<TNumber>>> opt_arr = view0;
  EXPECT_EQ(arr_nested.use_count(), 2);

  Optional<Array<Array<TNumber>>> arr1 = view0;
  EXPECT_EQ(arr_nested.use_count(), 3);
  EXPECT_EQ(arr1.value()[1][1].as<TFloatObj>()->value, 2);

  Any any1;
  Optional<Array<Array<TNumber>>> arr2 = any1;
  EXPECT_TRUE(arr2 == nullptr);

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] Optional<Array<Array<int>>> arr2 = view0;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("to `Optional<Array<Array<int>>>`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

}  // namespace
