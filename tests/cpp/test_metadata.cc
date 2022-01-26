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
#include <tvm/runtime/metadata.h>

namespace {
const int64_t kNormalInput1Shape[4] = {1, 5, 5, 3};
const struct TVMTensorInfo kNormalInputs[1] = {{"input1", kNormalInput1Shape, 4, DLDataType{1, 2, 3}}};

const int64_t kNormalOutput1Shape[3] = {3, 8, 8};
const struct TVMTensorInfo kNormalOutputs[1] = {{"output1", kNormalOutput1Shape, 3, DLDataType{3, 4, 5}}};

const char* kNormalDevices[2] = {"device1", "device2"};

const struct TVMMetadata kNormal = {
  TVM_METADATA_VERSION,
  kNormalInputs,
  1,
  kNormalOutputs,
  1,
  kNormalDevices,
  2,
  "aot",
  "default",
  "c",
  true,
  };
}

using ::testing::Eq;
using ::testing::ElementsAre;

TEST(Metadata, ParseStruct) {
  tvm::runtime::metadata::Metadata md = tvm::runtime::metadata::Metadata(&kNormal);
  EXPECT_THAT(md->version(), Eq(TVM_METADATA_VERSION));
  EXPECT_THAT(md->num_inputs(), Eq(1));

  auto input1 = md->inputs()[0];
  EXPECT_THAT(input1->name(), Eq("input1"));
  EXPECT_THAT(input1->shape(), ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(input1->dtype(), Eq(tvm::runtime::DataType(DLDataType{1, 2, 3})));
  // auto md_inputs = md->inputs();
  // EXPECT_EQ(md_inputs.size(), 1);

  // auto md_input = md_inputs[0];

  // EXPECT_EQ(md->get_name(), kNormal.name);
//  EXPECT_EQ(md->get_name(), kNormal.name);
}
