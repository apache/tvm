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
#include <llvm/IR/Intrinsics.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

using ::testing::HasSubstr;

// ---------
// Data Type
// ---------
TEST(TIR, TestCreateScalableType) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 4, true);
  ASSERT_EQ(scalable_type.code(), kDLInt);
  ASSERT_EQ(scalable_type.bits(), 32);
  ASSERT_EQ(scalable_type.lanes(), 4);
  ASSERT_TRUE(scalable_type.is_scalable());
  ASSERT_TRUE(scalable_type.is_vector());
}

TEST(TIR, TestScalableWithBits) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 1, 1, true);
  scalable_type = scalable_type.with_bits(32);
  ASSERT_EQ(scalable_type.bits(), 32);
  ASSERT_TRUE(scalable_type.is_scalable());
  ASSERT_TRUE(scalable_type.is_vector());
}

TEST(TIR, TestScalableWithLanes) {
  tvm::DataType type = tvm::DataType(kDLInt, 32, 1);
  tvm::DataType scalable_type = type.with_scalable_lanes(4);
  ASSERT_EQ(scalable_type.lanes(), 4);
  ASSERT_TRUE(scalable_type.is_scalable());
  ASSERT_TRUE(scalable_type.is_vector());
}

TEST(TIR, TestAssignScalableDataType) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 1, true);
  tvm::DataType scalable_type_copy = scalable_type;
  ASSERT_TRUE(scalable_type_copy.is_scalable());
  ASSERT_TRUE(scalable_type_copy.is_vector());
}

TEST(TIR, TestScalableDataTypeAndNonScalableDataTypeInequality) {
  ASSERT_FALSE(tvm::DataType(kDLInt, 32, 4, true) == tvm::DataType(kDLInt, 32, 4));
}

TEST(TIR, TestGetScalableVectorBytes) {
  tvm::DataType scalable_type = tvm::DataType(kDLInt, 32, 4, true);
  EXPECT_THROW(
      {
        try {
          tvm::runtime::GetVectorBytes(scalable_type);
        } catch (const tvm::InternalError& e) {
          EXPECT_THAT(e.what(), HasSubstr("Cannot get vector bytes of scalable vector"));
          throw;
        }
      },
      tvm::InternalError);
}