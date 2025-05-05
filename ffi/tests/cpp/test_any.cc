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
#include <tvm/ffi/memory.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Any, Int) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  std::optional<int64_t> opt_v0 = view0.TryAs<int64_t>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] int64_t v0 = view0;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `int`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  AnyView view1 = 1;
  EXPECT_EQ(view1.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIInt);
  EXPECT_EQ(view1.CopyToTVMFFIAny().v_int64, 1);

  int32_t int_v1 = view1;
  EXPECT_EQ(int_v1, 1);

  int64_t v1 = 2;
  view0 = v1;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIInt);
  EXPECT_EQ(view0.CopyToTVMFFIAny().v_int64, 2);
}

TEST(Any, Float) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  std::optional<double> opt_v0 = view0.TryAs<double>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] double v0 = view0;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `float`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  AnyView view1_int = 1;
  float float_v1 = view1_int;
  EXPECT_EQ(float_v1, 1);

  AnyView view2 = 2.2;
  EXPECT_EQ(view2.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIFloat);
  EXPECT_EQ(view2.CopyToTVMFFIAny().v_float64, 2.2);

  float v1 = 2;
  view0 = v1;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIFloat);
  EXPECT_EQ(view0.CopyToTVMFFIAny().v_float64, 2);
}

TEST(Any, Object) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  // int object is not nullable
  std::optional<TInt> opt_v0 = view0.TryAs<TInt>();
  EXPECT_TRUE(!opt_v0.has_value());

  TInt v1(11);
  EXPECT_EQ(v1.use_count(), 1);
  // view won't increase refcount
  AnyView view1 = v1;
  EXPECT_EQ(v1.use_count(), 1);
  // any will trigger ref count increase
  Any any1 = v1;
  EXPECT_EQ(v1.use_count(), 2);
  // copy to another view
  AnyView view2 = any1;
  EXPECT_EQ(v1.use_count(), 2);

  // convert to weak raw object ptr
  const TIntObj* v1_ptr = view2;
  EXPECT_EQ(v1.use_count(), 2);
  EXPECT_EQ(v1_ptr->value, 11);
  Any any2 = v1_ptr;
  EXPECT_EQ(v1.use_count(), 3);
  EXPECT_TRUE(any2.TryAs<TInt>().has_value());

  // convert to raw opaque ptr
  void* raw_v1_ptr = const_cast<TIntObj*>(v1_ptr);
  any2 = raw_v1_ptr;
  EXPECT_TRUE(any2.TryAs<void*>().value() == v1_ptr);

  // convert to ObjectPtr
  ObjectPtr<TNumberObj> v1_obj_ptr = view2;
  EXPECT_EQ(v1.use_count(), 3);
  any2 = v1_obj_ptr;
  EXPECT_EQ(v1.use_count(), 4);
  EXPECT_TRUE(any2.TryAs<TInt>().has_value());
  any2.reset();
  v1_obj_ptr.reset();

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] TFloat v0 = view1;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          std::cout << what;
          EXPECT_NE(what.find("Cannot convert from type `test.Int` to `test.Float`"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
  // Try to convert to number
  TNumber number0 = any1;
  EXPECT_EQ(v1.use_count(), 3);
  EXPECT_TRUE(number0.as<TIntObj>());
  EXPECT_EQ(number0.as<TIntObj>()->value, 11);
  EXPECT_TRUE(!any1.TryAs<int>().has_value());

  TInt int1 = view2;
  EXPECT_EQ(v1.use_count(), 4);
  any1.reset();
  EXPECT_EQ(v1.use_count(), 3);
}

}  // namespace
