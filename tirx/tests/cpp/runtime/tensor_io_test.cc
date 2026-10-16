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
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/bytes_io.h>

#include <cstring>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace {

std::string Save(const ffi::Tensor& tensor) {
  std::string buffer;
  support::BytesOutStream strm(&buffer);
  SaveTensor(tensor, &strm);
  return buffer;
}

ffi::Tensor Load(const std::string& buffer) {
  support::BytesInStream strm(buffer);
  return LoadTensor(&strm);
}

void ExpectRoundTrip(DLDataType dtype, std::vector<int64_t> shape) {
  ffi::Tensor tensor = Empty(ffi::Shape(shape), dtype, Device{kDLCPU, 0});
  size_t nbytes = ffi::GetDataSize(*tensor.GetDLTensorPtr());
  auto* data = static_cast<uint8_t*>(tensor.GetDLTensorPtr()->data);
  for (size_t i = 0; i < nbytes; ++i) data[i] = static_cast<uint8_t>(i * 7 + 1);

  ffi::Tensor loaded = Load(Save(tensor));
  const DLTensor* out = loaded.GetDLTensorPtr();
  EXPECT_EQ(out->dtype, dtype);
  ASSERT_EQ(out->ndim, static_cast<int>(shape.size()));
  for (size_t i = 0; i < shape.size(); ++i) EXPECT_EQ(out->shape[i], shape[i]);
  ASSERT_EQ(ffi::GetDataSize(*out), nbytes);
  EXPECT_EQ(std::memcmp(out->data, data, nbytes), 0);
}

}  // namespace

TEST(TensorIO, RoundTripScalarDType) { ExpectRoundTrip(DLDataType{kDLFloat, 32, 1}, {2, 3}); }

TEST(TensorIO, RoundTripVectorLanes) { ExpectRoundTrip(DLDataType{kDLFloat, 32, 4}, {5}); }

TEST(TensorIO, RoundTripSubByteDType) { ExpectRoundTrip(DLDataType{kDLInt, 4, 1}, {6}); }

TEST(TensorIO, JsonPreservesBase64Payload) {
  // Golden payloads from the stream encoding, covering both padding sizes and no padding.
  const char* expected[] = {
      "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAEIAQABAAAAAAAAAAEAAAAAAAAAAA==",
      "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAEIAQACAAAAAAAAAAIAAAAAAAAAAIA=",
      "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAEIAQADAAAAAAAAAAMAAAAAAAAAAID/",

  };
  auto to_json = ffi::reflection::TypeAttrColumn(
                     ffi::type_attr::kDataToJson)[ffi::TensorObj::RuntimeTypeIndex()]
                     .cast<ffi::Function>();
  auto from_json = ffi::reflection::TypeAttrColumn(
                       ffi::type_attr::kDataFromJson)[ffi::TensorObj::RuntimeTypeIndex()]
                       .cast<ffi::Function>();
  const uint8_t data[] = {0, 128, 255};
  for (int64_t size = 1; size <= 3; ++size) {
    ffi::Tensor tensor = Empty(ffi::Shape({size}), DLDataType{kDLUInt, 8, 1}, Device{kDLCPU, 0});
    std::memcpy(tensor.GetDLTensorPtr()->data, data, size);
    ffi::String encoded = to_json(tensor).cast<ffi::String>();
    EXPECT_EQ(encoded, expected[size - 1]);
    auto loaded = from_json(encoded).cast<ffi::Tensor>();
    EXPECT_EQ(loaded.dtype(), tensor.dtype());
    ASSERT_EQ(loaded.shape()[0], size);
    EXPECT_EQ(std::memcmp(loaded.GetDLTensorPtr()->data, data, size), 0);
  }
}

TEST(TensorIO, RejectsTruncatedData) {
  ffi::Tensor tensor = Empty(ffi::Shape({4}), DLDataType{kDLInt, 32, 1}, Device{kDLCPU, 0});
  std::memset(tensor.GetDLTensorPtr()->data, 0, ffi::GetDataSize(*tensor.GetDLTensorPtr()));
  std::string buffer = Save(tensor);
  buffer.pop_back();
  EXPECT_THROW(Load(buffer), ffi::Error);
}

}  // namespace runtime
}  // namespace tvm
