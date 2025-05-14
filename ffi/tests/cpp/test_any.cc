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

  Optional<int64_t> opt_v0 = view0.as<int64_t>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto v0 = view0.cast<int>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `int`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  AnyView view1 = 1;
  EXPECT_EQ(view1.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIInt);
  EXPECT_EQ(view1.CopyToTVMFFIAny().v_int64, 1);

  auto int_v1 = view1.cast<int>();
  EXPECT_EQ(int_v1, 1);

  int64_t v1 = 2;
  view0 = v1;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIInt);
  EXPECT_EQ(view0.CopyToTVMFFIAny().v_int64, 2);
}

TEST(Any, bool) {
  AnyView view0;
  Optional<bool> opt_v0 = view0.as<bool>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto v0 = view0.cast<bool>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `bool`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  AnyView view1 = true;
  EXPECT_EQ(view1.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIBool);
  EXPECT_EQ(view1.CopyToTVMFFIAny().v_int64, 1);

  auto int_v1 = view1.cast<int>();
  EXPECT_EQ(int_v1, 1);

  bool v1 = false;
  view0 = v1;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIBool);
  EXPECT_EQ(view0.CopyToTVMFFIAny().v_int64, 0);
}

TEST(Any, nullptrcmp) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);
  EXPECT_TRUE(view0 == nullptr);
  EXPECT_FALSE(view0 != nullptr);

  view0 = 1;
  EXPECT_TRUE(view0 != nullptr);
  EXPECT_FALSE(view0 == nullptr);

  Any any0 = view0;
  EXPECT_TRUE(any0 != nullptr);
  EXPECT_FALSE(any0 == nullptr);

  any0 = nullptr;
  EXPECT_TRUE(any0 == nullptr);
  EXPECT_FALSE(any0 != nullptr);
}

TEST(Any, Float) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  Optional<double> opt_v0 = view0.as<double>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto v0 = view0.cast<double>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `float`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  AnyView view1_int = 1;
  auto float_v1 = view1_int.cast<float>();
  EXPECT_EQ(float_v1, 1);

  AnyView view2 = 2.2;
  EXPECT_EQ(view2.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIFloat);
  EXPECT_EQ(view2.CopyToTVMFFIAny().v_float64, 2.2);

  float v1 = 2;
  view0 = v1;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFIFloat);
  EXPECT_EQ(view0.CopyToTVMFFIAny().v_float64, 2);
}

TEST(Any, Device) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  Optional<DLDevice> opt_v0 = view0.as<DLDevice>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto v0 = view0.cast<DLDevice>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `Device`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  DLDevice device{kDLCUDA, 1};

  AnyView view1_device = device;
  auto dtype_v1 = view1_device.cast<DLDevice>();
  EXPECT_EQ(dtype_v1.device_type, kDLCUDA);
  EXPECT_EQ(dtype_v1.device_id, 1);

  Any any2 = DLDevice{kDLCPU, 0};
  TVMFFIAny ffi_v2 = details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(any2));
  EXPECT_EQ(ffi_v2.type_index, TypeIndex::kTVMFFIDevice);
  EXPECT_EQ(ffi_v2.v_device.device_type, kDLCPU);
  EXPECT_EQ(ffi_v2.v_device.device_id, 0);
}

TEST(Any, DLTensor) {
  AnyView view0;

  Optional<DLTensor*> opt_v0 = view0.as<DLTensor*>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto v0 = view0.cast<DLTensor*>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `DLTensor*`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  DLTensor dltensor;

  AnyView view1_dl = &dltensor;
  auto dl_v1 = view1_dl.cast<DLTensor*>();
  EXPECT_EQ(dl_v1, &dltensor);
}

TEST(Any, Object) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  // int object is not nullable
  Optional<TInt> opt_v0 = view0.as<TInt>();
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
  const TIntObj* v1_ptr = view2.cast<const TIntObj*>();
  EXPECT_EQ(v1.use_count(), 2);
  EXPECT_EQ(v1_ptr->value, 11);
  Any any2 = v1_ptr;
  EXPECT_EQ(v1.use_count(), 3);
  EXPECT_TRUE(any2.as<TInt>().has_value());

  // convert to raw opaque ptr
  void* raw_v1_ptr = const_cast<TIntObj*>(v1_ptr);
  any2 = raw_v1_ptr;
  EXPECT_TRUE(any2.as<void*>().value() == v1_ptr);

  // convert to ObjectRef
  {
    auto v1_obj_ref = view2.cast<TNumber>();
    EXPECT_EQ(v1.use_count(), 3);
    any2 = v1_obj_ref;
    EXPECT_EQ(v1.use_count(), 4);
    EXPECT_TRUE(any2.as<TInt>().has_value());
    any2.reset();
  }

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto v0 = view1.cast<TFloat>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          std::cout << what;
          EXPECT_NE(what.find("Cannot convert from type `test.Int` to `test.Float`"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
  // Try to convert to number
  auto number0 = any1.cast<TNumber>();
  EXPECT_EQ(v1.use_count(), 3);
  EXPECT_TRUE(number0.as<TIntObj>());
  EXPECT_EQ(number0.as<TIntObj>()->value, 11);
  EXPECT_TRUE(!any1.as<int>().has_value());

  auto int1 = view2.cast<TInt>();
  EXPECT_EQ(v1.use_count(), 4);
  any1.reset();
  EXPECT_EQ(v1.use_count(), 3);
}

TEST(Any, ObjectRefWithFallbackTraits) {
  // Test case for TPrimExpr fallback from Any
  Any any1 = TPrimExpr("float32", 3.14);
  auto v0 = any1.cast<TPrimExpr>();
  EXPECT_EQ(v0->value, 3.14);
  EXPECT_EQ(v0->dtype, "float32");

  any1 = true;
  auto v1 = any1.cast<TPrimExpr>();
  EXPECT_EQ(v1->value, 1);
  EXPECT_EQ(v1->dtype, "bool");

  any1 = int64_t(42);
  auto v2 = any1.cast<TPrimExpr>();
  EXPECT_EQ(v2->value, 42);
  EXPECT_EQ(v2->dtype, "int64");

  any1 = 2.718;
  auto v3 = any1.cast<TPrimExpr>();
  EXPECT_EQ(v3->value, 2.718);
  EXPECT_EQ(v3->dtype, "float32");

  // Test case for TPrimExpr fallback from AnyView
  TPrimExpr texpr1("float32", 3.14);
  AnyView view1 = texpr1;
  auto v4 = view1.cast<TPrimExpr>();
  EXPECT_EQ(v4->value, 3.14);
  EXPECT_EQ(v4->dtype, "float32");

  view1 = true;
  auto v5 = view1.cast<TPrimExpr>();
  EXPECT_EQ(v5->value, 1);
  EXPECT_EQ(v5->dtype, "bool");

  view1 = int64_t(42);
  auto v6 = view1.cast<TPrimExpr>();
  EXPECT_EQ(v6->value, 42);
  EXPECT_EQ(v6->dtype, "int64");

  view1 = 2.718;
  auto v7 = view1.cast<TPrimExpr>();
  EXPECT_EQ(v7->value, 2.718);
  EXPECT_EQ(v7->dtype, "float32");

  // Test case for TPrimExpr fallback from Any with String
  any1 = std::string("test_string");
  auto v8 = any1.cast<TPrimExpr>();
  EXPECT_EQ(v8->dtype, "test_string");
  EXPECT_EQ(v8->value, 0);

  // Test case for TPrimExpr fallback from AnyView with String
  view1 = "test_string";
  auto v9 = view1.cast<TPrimExpr>();
  EXPECT_EQ(v9->dtype, "test_string");
  EXPECT_EQ(v9->value, 0);
}

TEST(Any, CastVsAs) {
  AnyView view0 = 1;
  // as only runs strict check
  auto opt_v0 = view0.as<int64_t>();
  EXPECT_TRUE(opt_v0.has_value());
  EXPECT_EQ(opt_v0.value(), 1);

  auto opt_v1 = view0.as<bool>();
  EXPECT_TRUE(!opt_v1.has_value());
  auto opt_v2 = view0.as<double>();
  EXPECT_TRUE(!opt_v2.has_value());

  // try_cast will try run the conversion.
  auto opt_v3 = view0.try_cast<bool>();
  EXPECT_TRUE(opt_v3.has_value());
  EXPECT_EQ(opt_v3.value(), 1);
  auto opt_v4 = view0.try_cast<double>();
  EXPECT_TRUE(opt_v4.has_value());
  EXPECT_EQ(opt_v4.value(), 1);

  Any any1 = true;
  auto opt_v5 = any1.as<bool>();
  EXPECT_TRUE(opt_v5.has_value());
  EXPECT_EQ(opt_v5.value(), 1);

  auto opt_v6 = any1.try_cast<int>();
  EXPECT_TRUE(opt_v6.has_value());
  EXPECT_EQ(opt_v6.value(), 1);

  auto opt_v7 = any1.try_cast<double>();
  EXPECT_TRUE(opt_v7.has_value());
}

TEST(Any, ObjectMove) {
  Any any1 = TPrimExpr("float32", 3.14);
  auto v0 = std::move(any1).cast<TPrimExpr>();
  EXPECT_EQ(v0->value, 3.14);
  EXPECT_EQ(v0.use_count(), 1);
  EXPECT_TRUE(any1 == nullptr);
}

}  // namespace
