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
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/memory.h>

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(DType, StringConversion) {
  DLDataType dtype = DLDataType{kDLFloat, 32, 1};
  EXPECT_EQ(DLDataTypeToString(dtype), "float32");
  EXPECT_EQ(StringToDLDataType("float32"), dtype);

  dtype = DLDataType{kDLInt, 16, 2};
  EXPECT_EQ(DLDataTypeToString(dtype), "int16x2");
  EXPECT_EQ(StringToDLDataType("int16x2"), dtype);

  dtype = DLDataType{kDLOpaqueHandle, 0, 0};
  EXPECT_EQ(DLDataTypeToString(dtype), "void");
  EXPECT_EQ(StringToDLDataType("void"), dtype);

  // test bfloat with lanes
  dtype = DLDataType{kDLBfloat, 16, 2};
  EXPECT_EQ(DLDataTypeToString(dtype), "bfloat16x2");
  EXPECT_EQ(StringToDLDataType("bfloat16x2"), dtype);

  // test float8
  dtype = DLDataType{kDLExtFloat8_e4m3fn, 8, 2};
  EXPECT_EQ(DLDataTypeToString(dtype), "float8_e4m3fnx2");
  EXPECT_EQ(StringToDLDataType("float8_e4m3fnx2"), dtype);
}

TEST(DataType, AnyConversion) {
  AnyView view0;
  EXPECT_EQ(view0.CopyToTVMFFIAny().type_index, TypeIndex::kTVMFFINone);

  std::optional<DLDataType> opt_v0 = view0.as<DLDataType>();
  EXPECT_TRUE(!opt_v0.has_value());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] DLDataType v0 = view0;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `None` to `DataType`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  DLDataType dtype{kDLFloat, 32, 1};

  AnyView view1_dtype = dtype;
  DLDataType dtype_v1 = view1_dtype;
  EXPECT_EQ(dtype_v1.code, kDLFloat);
  EXPECT_EQ(dtype_v1.bits, 32);
  EXPECT_EQ(dtype_v1.lanes, 1);

  Any view2 = DLDataType{kDLInt, 16, 2};
  TVMFFIAny ffi_v2;
  view2.MoveToTVMFFIAny(&ffi_v2);
  EXPECT_EQ(ffi_v2.type_index, TypeIndex::kTVMFFIDataType);
  EXPECT_EQ(ffi_v2.v_dtype.code, kDLInt);
  EXPECT_EQ(ffi_v2.v_dtype.bits, 16);
  EXPECT_EQ(ffi_v2.v_dtype.lanes, 2);
}

// String can be automatically converted to DLDataType
TEST(DataType, AnyConversionWithString) {
  AnyView view0 = "float32";

  std::optional<DLDataType> opt_v0 = view0.as<DLDataType>();
  DLDataType dtype_v0 = opt_v0.value();
  EXPECT_EQ(dtype_v0.code, kDLFloat);
  EXPECT_EQ(dtype_v0.bits, 32);
  EXPECT_EQ(dtype_v0.lanes, 1);

  Any any = String("bfloat16x2");
  std::optional<DLDataType> opt_v1 = any.as<DLDataType>();
  EXPECT_EQ(opt_v1.value().code, kDLBfloat);
  EXPECT_EQ(opt_v1.value().bits, 16);
  EXPECT_EQ(opt_v1.value().lanes, 2);
}
}  // namespace
