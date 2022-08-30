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

#include "../../../src/support/scalars.h"

#include <gtest/gtest.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace support {
namespace {

// Note that functional testing is via test_ir_parser.py and test_ir_text_printer.py.
// Here we just check handling which is difficult to test via the standard Python API.

TEST(Scalars, IntImmToNDArray_Unsupported) {
  ASSERT_THROW(IntImmToNDArray(IntImm(DataType::Int(15), 42)), runtime::InternalError);
}

TEST(Scalars, FloatImmtoNDArray_Unsupported) {
  ASSERT_THROW(FloatImmToNDArray(FloatImm(DataType::Float(15), 42.0)), runtime::InternalError);
}

TEST(Scalars, NDArrayScalarToString_Unsupported) {
  auto ndarray = runtime::NDArray::Empty({}, DataType::Int(8), {DLDeviceType::kDLCPU, 0});
  ASSERT_THROW(NDArrayScalarToString(ndarray), runtime::InternalError);
}

TEST(Scalars, IntImmToString_Unsupported) {
  ASSERT_THROW(IntImmToString(IntImm(DataType::Int(15), 42)), runtime::InternalError);
}

TEST(Scalars, FloatImmToString_Unsupported) {
  ASSERT_THROW(FloatImmToString(FloatImm(DataType::Float(15), 42.0)), runtime::InternalError);
}

TEST(Scalars, ValueToIntImm_Unsupported) {
  ASSERT_THROW(ValueToIntImm(42, 15), runtime::InternalError);
}

TEST(SCalars, ValueToFloatImm_Unsupported) {
  ASSERT_THROW(ValueToFloatImm(42.0, 15), runtime::InternalError);
}

}  // namespace
}  // namespace support
}  // namespace tvm
