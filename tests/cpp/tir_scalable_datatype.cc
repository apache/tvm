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
#include <tvm/runtime/data_type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

#ifdef TVM_LLVM_VERSION
#include <llvm/IR/Intrinsics.h>
#endif

#include "../../src/script/printer/utils.h"

using ::testing::HasSubstr;

// ---------
// Data Type
// ---------
TEST(ScalableDataType, TestCreateScalableType) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 4, true);
  ASSERT_EQ(scalable_type.code(), kDLInt);
  ASSERT_EQ(scalable_type.bits(), 32);
  ASSERT_EQ(scalable_type.vscale_factor(), 4);
  ASSERT_TRUE(scalable_type.is_scalable_vector());
  ASSERT_TRUE(scalable_type.is_scalable_or_fixed_length_vector());
}

TEST(ScalableDataType, TestScalableWithBits) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 1, 8, true);
  scalable_type = scalable_type.with_bits(32);
  ASSERT_EQ(scalable_type.bits(), 32);
  ASSERT_TRUE(scalable_type.is_scalable_vector());
  ASSERT_TRUE(scalable_type.is_scalable_or_fixed_length_vector());
}

TEST(ScalableDataType, TestScalableWithVscaleFactor) {
  tvm::DataType type = tvm::DataType(kDLInt, 32, 1);
  tvm::DataType scalable_type = type.with_scalable_vscale_factor(4);
  ASSERT_EQ(scalable_type.vscale_factor(), 4);
  ASSERT_TRUE(scalable_type.is_scalable_vector());
  ASSERT_TRUE(scalable_type.is_scalable_or_fixed_length_vector());
}

TEST(ScalableDataType, TestAssignScalableDataType) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 2, true);
  tvm::DataType scalable_type_copy = scalable_type;
  ASSERT_TRUE(scalable_type_copy.is_scalable_vector());
  ASSERT_TRUE(scalable_type_copy.is_scalable_or_fixed_length_vector());
}

TEST(ScalableDataType, TestScalableDataTypeEquality) {
  ASSERT_TRUE(tvm::DataType(kDLInt, 32, 4, true) == tvm::DataType(kDLInt, 32, 4, true));
}

TEST(ScalableDataType, TestScalableDataTypeAndNonScalableDataTypeInequality) {
  ASSERT_FALSE(tvm::DataType(kDLInt, 32, 4, true) == tvm::DataType(kDLInt, 32, 4));
}

TEST(ScalableDataType, TestIsScalar) {
  ASSERT_FALSE(tvm::DataType(kDLInt, 32, 4, true).is_scalar());
  ASSERT_TRUE(tvm::DataType(kDLInt, 32, 1, false).is_scalar());
  ASSERT_FALSE(tvm::DataType(kDLInt, 32, 4, false).is_scalar());
  ASSERT_FALSE(tvm::DataType(kDLOpaqueHandle, 1, 0, false).is_scalar());
}

TEST(ScalableDataType, TestScalableDataTypeToString) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 4, true);
  EXPECT_EQ(tvm::runtime::DLDataType2String(scalable_type), "int32xvscalex4");
}

TEST(ScalableDataType, TestStringToScalableDataType) {
  std::string scalable_type_str = "int32xvscalex4";
  EXPECT_EQ(tvm::DataType(tvm::runtime::String2DLDataType(scalable_type_str)),
            tvm::DataType(kDLInt, 32, 4, true));
}

TEST(ScalableDataType, TestInvalidStringToScalableDataType) {
  std::string scalable_type_str = "int32x4xvscale";
  EXPECT_THROW(
      {
        try {
          tvm::runtime::String2DLDataType(scalable_type_str);
        } catch (const tvm::InternalError& e) {
          EXPECT_THAT(e.what(), HasSubstr("unknown type int32x4xvscale"));
          throw;
        }
      },
      tvm::InternalError);
}

TEST(ScalableDataType, TestGetScalableVectorBytes) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 4, true);
  EXPECT_THROW(
      {
        try {
          tvm::runtime::GetVectorBytes(scalable_type);
        } catch (const tvm::InternalError& e) {
          EXPECT_THAT(e.what(),
                      HasSubstr("Can't fetch the lanes of a scalable vector at a compile time"));
          throw;
        }
      },
      tvm::InternalError);
}

TEST(ScalableDataType, TestScalableDataTypeInvalidLanesError) {
  EXPECT_THROW(
      {
        try {
          tvm::DataType(kDLFloat, 62, 1, true);
        } catch (const tvm::InternalError& e) {
          EXPECT_THAT(e.what(), HasSubstr("Invalid value for vscale factor"));
          throw;
        }
      },
      tvm::InternalError);
}

TEST(ScalableDataType, TestScalableDataTypeInvalidVscaleFactorAccess) {
  tvm::DataType fixed_length_type = tvm::DataType(kDLFloat, 32, 4);
  ASSERT_TRUE(fixed_length_type.is_fixed_length_vector());
  ASSERT_TRUE(fixed_length_type.is_scalable_or_fixed_length_vector());
  EXPECT_THROW(
      {
        try {
          fixed_length_type.vscale_factor();
        } catch (const tvm::InternalError& e) {
          EXPECT_THAT(e.what(), HasSubstr("A fixed length vector doesn't have a vscale factor"));
          throw;
        }
      },
      tvm::InternalError);
}

TEST(ScalableDataType, TestScalableDataTypeInvalidLanesAccess) {
  tvm::DataType scalable_type = tvm::DataType(kDLFloat, 32, 4, true);
  EXPECT_THROW(
      {
        try {
          scalable_type.lanes();
        } catch (const tvm::InternalError& e) {
          EXPECT_THAT(e.what(),
                      HasSubstr("Can't fetch the lanes of a scalable vector at a compile time"));
          throw;
        }
      },
      tvm::InternalError);
}

TEST(ScalableDataType, TestScalableBool) {
  tvm::DataType scalable_type = tvm::DataType::Bool(4, true);
  ASSERT_EQ(scalable_type.code(), kDLUInt);
  ASSERT_EQ(scalable_type.bits(), 1);
  ASSERT_EQ(scalable_type.vscale_factor(), 4);
  ASSERT_TRUE(scalable_type.is_scalable_vector());
}

TEST(ScalableDataType, TestScalableUInt) {
  tvm::DataType scalable_type = tvm::DataType::UInt(1, 4, true);
  ASSERT_EQ(scalable_type.code(), kDLUInt);
  ASSERT_EQ(scalable_type.bits(), 1);
  ASSERT_EQ(scalable_type.vscale_factor(), 4);
  ASSERT_TRUE(scalable_type.is_scalable_vector());
}

// -----------
// Integration
// -----------
#if TVM_LLVM_VERSION >= 130
TEST(ScalableDataType, TestScalableIntrinCall) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 4, true);
  tvm::tir::Call call = tvm::tir::Call(
      scalable_type, tvm::tir::builtin::call_llvm_intrin(),
      {tvm::IntImm(tvm::DataType::Int(32), ::llvm::Intrinsic::experimental_stepvector)});
  ASSERT_EQ(call->dtype, scalable_type);
  ASSERT_EQ(call->Script(),
            "T.call_llvm_intrin(\"int32xvscalex4\", \"llvm.experimental.stepvector\")");
}
#endif

TEST(ScalableDataType, TestTIRScriptScalableDtype2Str) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 4, true);
  ASSERT_EQ(tvm::script::printer::DType2Str(scalable_type), "int32xvscalex4");
}
