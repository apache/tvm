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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tvm/ffi/dtype.h>
#include <tvm/script/printer/printer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>

#ifdef TVM_LLVM_VERSION
#include <llvm/IR/Intrinsics.h>
#endif

#include "../../src/script/printer/utils.h"

using ::testing::HasSubstr;

// ---------
// Prim Type
// ---------
TEST(ScalablePrimType, TestCreateScalableType) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLInt, 32, 4);
  ASSERT_EQ(scalable_type.code(), kDLInt);
  ASSERT_EQ(scalable_type.bits(), 32);
  ASSERT_EQ(scalable_type.VScaleFactor(), 4);
  ASSERT_TRUE(scalable_type.IsScalableVector());
  ASSERT_TRUE(scalable_type.IsScalableVector() || scalable_type.IsFixedLengthVector());
}

TEST(ScalablePrimType, TestScalableWithBits) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLInt, 1, 8);
  scalable_type = scalable_type.WithBits(32);
  ASSERT_EQ(scalable_type.bits(), 32);
  ASSERT_TRUE(scalable_type.IsScalableVector());
  ASSERT_TRUE(scalable_type.IsScalableVector() || scalable_type.IsFixedLengthVector());
}

TEST(ScalablePrimType, TestScalableWithVscaleFactor) {
  tvm::PrimType type = tvm::PrimType::Int(32);
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(type.code(), type.bits(), 4);
  ASSERT_EQ(scalable_type.VScaleFactor(), 4);
  ASSERT_TRUE(scalable_type.IsScalableVector());
  ASSERT_TRUE(scalable_type.IsScalableVector() || scalable_type.IsFixedLengthVector());
}

TEST(ScalablePrimType, TestAssignScalablePrimType) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLInt, 32, 2);
  tvm::PrimType scalable_type_copy = scalable_type;
  ASSERT_TRUE(scalable_type_copy.IsScalableVector());
  ASSERT_TRUE(scalable_type_copy.IsScalableVector() || scalable_type_copy.IsFixedLengthVector());
}

TEST(ScalablePrimType, TestScalablePrimTypeEquality) {
  ASSERT_TRUE(tvm::PrimType::ScalableVector(kDLInt, 32, 4) ==
              tvm::PrimType::ScalableVector(kDLInt, 32, 4));
}

TEST(ScalablePrimType, TestScalablePrimTypeAndNonScalablePrimTypeInequality) {
  ASSERT_FALSE(tvm::PrimType::ScalableVector(kDLInt, 32, 4) == tvm::PrimType::Int(32, 4));
}

TEST(ScalablePrimType, TestIsScalar) {
  ASSERT_FALSE(tvm::PrimType::ScalableVector(kDLInt, 32, 4).IsScalar());
  ASSERT_TRUE(tvm::PrimType::Int(32).IsScalar());
  ASSERT_FALSE(tvm::PrimType::Int(32, 4).IsScalar());
  ASSERT_FALSE(tvm::PrimType::Void().IsScalar());
}

TEST(ScalablePrimType, TestScalablePrimTypeToString) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLInt, 32, 4);
  EXPECT_EQ(tvm::ffi::DLDataTypeToString(scalable_type->dtype), "int32xvscalex4");
}

TEST(ScalablePrimType, TestStringToScalablePrimType) {
  std::string scalable_type_str = "int32xvscalex4";
  EXPECT_EQ(tvm::PrimType(tvm::ffi::StringToDLDataType(scalable_type_str)),
            tvm::PrimType::ScalableVector(kDLInt, 32, 4));
}

TEST(ScalablePrimType, TestInvalidStringToScalablePrimType) {
  std::string scalable_type_str = "int32x4xvscale";
  EXPECT_THROW(
      {
        try {
          tvm::ffi::StringToDLDataType(scalable_type_str);
        } catch (const tvm::ffi::Error& e) {
          EXPECT_THAT(e.what(), HasSubstr("unknown dtype `int32x4xvscale`"));
          throw;
        }
      },
      tvm::ffi::Error);
}

TEST(ScalablePrimType, TestGetScalableVectorBytes) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLInt, 32, 4);
  EXPECT_THROW(
      {
        try {
          int bytes = (scalable_type.bits() * scalable_type.lanes() + 7) / 8;
          static_cast<void>(bytes);
        } catch (const tvm::ffi::Error& e) {
          EXPECT_THAT(e.what(),
                      HasSubstr("Can't fetch the lanes of a scalable vector at a compile time"));
          throw;
        }
      },
      tvm::ffi::Error);
}

TEST(ScalablePrimType, TestScalablePrimTypeInvalidLanesError) {
  EXPECT_THROW(
      {
        try {
          tvm::PrimType::ScalableVector(kDLFloat, 62, 1);
        } catch (const tvm::ffi::Error& e) {
          EXPECT_THAT(e.what(), HasSubstr("Invalid value for vscale factor"));
          throw;
        }
      },
      tvm::ffi::Error);
}

TEST(ScalablePrimType, TestScalablePrimTypeInvalidVscaleFactorAccess) {
  tvm::PrimType fixed_length_type = tvm::PrimType::Float(32, 4);
  ASSERT_TRUE(fixed_length_type.IsFixedLengthVector());
  ASSERT_TRUE(fixed_length_type.IsScalableVector() || fixed_length_type.IsFixedLengthVector());
  EXPECT_THROW(
      {
        try {
          fixed_length_type.VScaleFactor();
        } catch (const tvm::ffi::Error& e) {
          EXPECT_THAT(e.what(), HasSubstr("A fixed length vector doesn't have a vscale factor"));
          throw;
        }
      },
      tvm::ffi::Error);
}

TEST(ScalablePrimType, TestScalablePrimTypeInvalidLanesAccess) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLFloat, 32, 4);
  EXPECT_THROW(
      {
        try {
          scalable_type.lanes();
        } catch (const tvm::ffi::Error& e) {
          EXPECT_THAT(e.what(),
                      HasSubstr("Can't fetch the lanes of a scalable vector at a compile time"));
          throw;
        }
      },
      tvm::ffi::Error);
}

TEST(ScalablePrimType, TestScalableBool) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLBool, 8, 4);
  ASSERT_EQ(scalable_type.code(), kDLBool);
  ASSERT_EQ(scalable_type.bits(), 8);
  ASSERT_EQ(scalable_type.VScaleFactor(), 4);
  ASSERT_TRUE(scalable_type.IsScalableVector());
}

TEST(ScalablePrimType, TestScalableUInt) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLUInt, 1, 4);
  ASSERT_EQ(scalable_type.code(), kDLUInt);
  ASSERT_EQ(scalable_type.bits(), 1);
  ASSERT_EQ(scalable_type.VScaleFactor(), 4);
  ASSERT_TRUE(scalable_type.IsScalableVector());
}

// -----------
// Integration
// -----------
#ifdef TVM_LLVM_VERSION
TEST(ScalablePrimType, TestScalableIntrinCall) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLInt, 32, 4);
  tvm::Call call = tvm::Call(scalable_type, tvm::tirx::builtin::call_llvm_intrin(),
#if TVM_LLVM_VERSION >= 200
                             {tvm::IntImm::Int32(::llvm::Intrinsic::stepvector)});
#else
                             {tvm::IntImm::Int32(::llvm::Intrinsic::experimental_stepvector)});
#endif
  ASSERT_EQ(call->ty.as_or_throw<tvm::PrimType>(), scalable_type);
  ASSERT_EQ(tvm::Script(call),
#if TVM_LLVM_VERSION >= 200
            "T.call_llvm_intrin(\"int32xvscalex4\", \"llvm.stepvector\")");
#else
            "T.call_llvm_intrin(\"int32xvscalex4\", \"llvm.experimental.stepvector\")");
#endif
}
#endif

TEST(ScalablePrimType, TestTIRScriptScalableDtype2Str) {
  tvm::PrimType scalable_type = tvm::PrimType::ScalableVector(kDLInt, 32, 4);
  ASSERT_EQ(tvm::script::printer::DType2Str(scalable_type->dtype), "int32xvscalex4");
}
