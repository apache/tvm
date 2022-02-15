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
#include <tvm/ir/type.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/metadata.h>

namespace {
const int64_t kNormalInput1Shape[4] = {1, 5, 5, 3};
const struct TVMTensorInfo kNormalInputs[1] = {
    {"input1", kNormalInput1Shape, 4, DLDataType{1, 2, 3}}};

const int64_t kNormalOutput1Shape[3] = {3, 8, 8};
const struct TVMTensorInfo kNormalOutputs[1] = {
    {"output1", kNormalOutput1Shape, 3, DLDataType{3, 4, 5}}};

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
}  // namespace

using ::testing::ElementsAre;
using ::testing::Eq;
using ::tvm::runtime::Downcast;

TEST(Metadata, ParseStruct) {
  tvm::runtime::metadata::Metadata md = tvm::runtime::metadata::Metadata(&kNormal);
  EXPECT_THAT(md->version(), Eq(TVM_METADATA_VERSION));
  EXPECT_THAT(md->num_inputs(), Eq(1));

  auto input1 = md->inputs()[0];
  EXPECT_THAT(input1->name(), Eq("input1"));
  EXPECT_THAT(input1->shape(), ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(input1->dtype(), Eq(tvm::runtime::DataType(DLDataType{1, 2, 3})));

  EXPECT_THAT(md->num_outputs(), Eq(1));
  auto output1 = md->outputs()[0];
  EXPECT_THAT(output1->name(), Eq("output1"));
  EXPECT_THAT(::std::vector<int64_t>(output1->shape()), ElementsAre(3, 8, 8));
  EXPECT_THAT(output1->dtype(), Eq(tvm::runtime::DataType(DLDataType{3, 4, 5})));

  auto devices = md->devices();
  EXPECT_THAT(devices,
              ElementsAre(::tvm::runtime::String("device1"), ::tvm::runtime::String("device2")));

  EXPECT_THAT(md->executor(), Eq("aot"));
  EXPECT_THAT(md->mod_name(), Eq("default"));
  EXPECT_THAT(md->interface_api(), Eq("c"));
  EXPECT_THAT(md->use_unpacked_api(), Eq(true));
}

class TestVisitor : public tvm::AttrVisitor {
 public:
  using Element = ::std::tuple<::std::string, ::tvm::runtime::ObjectRef>;
  void Visit(const char* key, double* value) final {
    keys.push_back(key);
    values.push_back(::tvm::FloatImm(::tvm::runtime::DataType(kDLFloat, 64, 1), *value));
  }
  void Visit(const char* key, int64_t* value) final {
    keys.push_back(key);
    values.push_back(::tvm::IntImm(::tvm::runtime::DataType(kDLInt, 64, 1), *value));
  }
  void Visit(const char* key, uint64_t* value) final {
    keys.push_back(key);
    int64_t v;
    *(reinterpret_cast<uint64_t*>(&v)) = *value;
    values.push_back(::tvm::IntImm(::tvm::runtime::DataType(kDLUInt, 64, 1), v));
  }
  void Visit(const char* key, int* value) final {
    keys.push_back(key);
    values.push_back(::tvm::IntImm(::tvm::runtime::DataType(kDLInt, 64, 1), *value));
  }
  void Visit(const char* key, bool* value) final {
    keys.push_back(key);
    values.push_back(::tvm::Bool(*value));
  }
  void Visit(const char* key, std::string* value) final {
    keys.push_back(key);
    values.push_back(::tvm::runtime::String(*value));
  }
  void Visit(const char* key, tvm::runtime::DataType* value) final {
    keys.push_back(key);
    values.push_back(::tvm::PrimType(*value));
  }
  void Visit(const char* key, tvm::runtime::NDArray* value) final {
    keys.push_back(key);
    values.push_back(*value);
  }
  void Visit(const char* key, void** value) final { CHECK(false) << "Do not expect this type"; }

  void Visit(const char* key, ::tvm::runtime::ObjectRef* value) final {
    keys.push_back(key);
    values.push_back(*value);
  }

  std::vector<std::string> keys;
  std::vector<::tvm::runtime::ObjectRef> values;
};

TEST(Metadata, Visitor) {
  tvm::runtime::metadata::Metadata md = tvm::runtime::metadata::Metadata(&kNormal);
  TestVisitor v;
  ::tvm::ReflectionVTable::Global()->VisitAttrs(md.operator->(), &v);

  EXPECT_THAT(v.keys,
              ElementsAre(Eq("version"), Eq("inputs"), Eq("outputs"), Eq("devices"), Eq("executor"),
                          Eq("mod_name"), Eq("interface_api"), Eq("use_unpacked_api")));

  EXPECT_THAT(Downcast<tvm::IntImm>(v.values[0])->value, Eq(TVM_METADATA_VERSION));

  // Just identify the tensor.
  auto input_array = Downcast<tvm::runtime::metadata::MetadataArray>(v.values[1]);
  EXPECT_THAT(input_array->type_index, Eq(tvm::runtime::metadata::MetadataTypeIndex::kMetadata));
  EXPECT_THAT(input_array->struct_name, Eq(std::string("TVMTensorInfo")));
  EXPECT_THAT(input_array->array.size(), Eq(1));
  auto array0 = input_array->array[0];

  auto input1 = Downcast<tvm::runtime::metadata::TensorInfo>(array0);
  EXPECT_THAT(input1->name(), Eq("input1"));

  auto output_array = Downcast<tvm::runtime::metadata::MetadataArray>(v.values[2]);
  EXPECT_THAT(output_array->type_index, Eq(tvm::runtime::metadata::MetadataTypeIndex::kMetadata));
  EXPECT_THAT(output_array->struct_name, Eq("TVMTensorInfo"));
  auto output1 = Downcast<tvm::runtime::metadata::TensorInfo>(output_array->array[0]);

  EXPECT_THAT(output1->name(), Eq("output1"));

  auto devices = Downcast<tvm::runtime::metadata::MetadataArray>(v.values[3]);
  EXPECT_THAT(devices->type_index, Eq(tvm::runtime::metadata::MetadataTypeIndex::kString));
  EXPECT_THAT(Downcast<tvm::runtime::String>(devices->array[0]), Eq("device1"));
  EXPECT_THAT(Downcast<tvm::runtime::String>(devices->array[1]), Eq("device1"));

  //  EXPECT_THAT(Downcast<IntImm>(v.values[0])->value, Eq(TVM_METADATA_VERSION));
}
