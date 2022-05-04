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

#include "../src/target/metadata.h"
#include "../src/target/metadata_utils.h"

namespace {

const int64_t kNormalInput1Shape[4] = {1, 5, 5, 3};
const struct TVMTensorInfo kNormalInputs[2] = {
    {"input1", kNormalInput1Shape, 4, DLDataType{1, 2, 3}},
    {"input2", kNormalInput1Shape, 4, DLDataType{2, 3, 4}}};

const int64_t kNormalOutput1Shape[3] = {3, 8, 8};
const struct TVMTensorInfo kNormalOutputs[1] = {
    {"output1", kNormalOutput1Shape, 3, DLDataType{3, 4, 5}}};

const int64_t kNormalPool1Shape[3] = {3, 8, 8};
const struct TVMTensorInfo kNormalPools[1] = {{"pool1", kNormalPool1Shape, 3, DLDataType{3, 4, 7}}};

const struct TVMMetadata kNormal = {
    TVM_METADATA_VERSION, kNormalInputs, 2, kNormalOutputs, 1, kNormalPools, 1, "default",
};
}  // namespace

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;
using ::testing::StrEq;

using ::tvm::codegen::metadata::DiscoverArraysVisitor;
using ::tvm::codegen::metadata::DiscoverComplexTypesVisitor;
using ::tvm::codegen::metadata::kMetadataGlobalSymbol;

using ::tvm::runtime::Array;
using ::tvm::runtime::Downcast;
using ::tvm::runtime::ObjectRef;

using ::tvm::runtime::metadata::Metadata;
using ::tvm::runtime::metadata::MetadataArray;
using ::tvm::runtime::metadata::MetadataKind;
using ::tvm::runtime::metadata::TensorInfo;

TEST(Metadata, ParseStruct) {
  Metadata md = Metadata(&kNormal);
  EXPECT_THAT(md->version(), Eq(TVM_METADATA_VERSION));
  EXPECT_THAT(md->num_inputs(), Eq(2));

  auto inputs = md->inputs();
  EXPECT_THAT(inputs.size(), Eq(2));

  auto input1 = inputs[0];
  EXPECT_THAT(input1->name(), Eq("input1"));
  EXPECT_THAT(input1->shape(), ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(input1->dtype(), Eq(tvm::runtime::DataType(DLDataType{1, 2, 3})));

  auto input2 = inputs[1];
  EXPECT_THAT(input2->name(), Eq("input2"));
  EXPECT_THAT(input2->shape(), ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(input2->dtype(), Eq(tvm::runtime::DataType(DLDataType{2, 3, 4})));

  EXPECT_THAT(md->num_outputs(), Eq(1));
  auto outputs = md->outputs();
  EXPECT_THAT(outputs.size(), Eq(1));

  auto output1 = outputs[0];
  EXPECT_THAT(output1->name(), Eq("output1"));
  EXPECT_THAT(output1->shape(), ElementsAre(3, 8, 8));
  EXPECT_THAT(output1->dtype(), Eq(tvm::runtime::DataType(DLDataType{3, 4, 5})));

  auto pools = md->pools();
  EXPECT_THAT(pools.size(), Eq(1));

  auto pool1 = pools[0];
  EXPECT_THAT(pool1->name(), Eq("pool1"));
  EXPECT_THAT(pool1->shape(), ElementsAre(3, 8, 8));
  EXPECT_THAT(pool1->dtype(), Eq(tvm::runtime::DataType(DLDataType{3, 4, 7})));

  EXPECT_THAT(md->mod_name(), Eq("default"));
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
  Metadata md = Metadata(&kNormal);
  TestVisitor v;
  ::tvm::ReflectionVTable::Global()->VisitAttrs(md.operator->(), &v);

  EXPECT_THAT(v.keys, ElementsAre(StrEq("version"), StrEq("inputs"), StrEq("num_inputs"),
                                  StrEq("outputs"), StrEq("num_outputs"), StrEq("pools"),
                                  StrEq("num_pools"), StrEq("mod_name")));
  EXPECT_THAT(Downcast<tvm::IntImm>(v.values[0])->value, Eq(TVM_METADATA_VERSION));

  EXPECT_THAT(Downcast<tvm::IntImm>(v.values[0])->value, Eq(TVM_METADATA_VERSION));

  // Just identify the tensor.
  auto input_array = Downcast<MetadataArray>(v.values[1]);
  EXPECT_THAT(input_array->kind, Eq(MetadataKind::kMetadata));
  EXPECT_THAT(input_array->type_key, StrEq("metadata.TensorInfoNode"));
  EXPECT_THAT(input_array->array.size(), Eq(2));

  auto input1 = Downcast<TensorInfo>(input_array->array[0]);
  EXPECT_THAT(input1->name(), StrEq("input1"));
  EXPECT_THAT(input1->shape(), ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(input1->dtype(), tvm::runtime::DataType(DLDataType{1, 2, 3}));

  auto input2 = Downcast<TensorInfo>(input_array->array[1]);
  EXPECT_THAT(input1->name(), StrEq("input1"));
  EXPECT_THAT(input1->shape(), ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(input1->dtype(), tvm::runtime::DataType(DLDataType{1, 2, 3}));

  auto num_inputs = Downcast<tvm::IntImm>(v.values[2]);
  EXPECT_THAT(num_inputs->value, Eq(2));

  auto output_array = Downcast<MetadataArray>(v.values[3]);
  EXPECT_THAT(output_array->kind, Eq(MetadataKind::kMetadata));
  EXPECT_THAT(output_array->type_key, StrEq("metadata.TensorInfoNode"));
  auto output1 = Downcast<TensorInfo>(output_array->array[0]);

  EXPECT_THAT(output1->name(), Eq("output1"));

  auto num_outputs = Downcast<tvm::IntImm>(v.values[4]);
  EXPECT_THAT(num_outputs->value, Eq(1));

  auto pool_array = Downcast<MetadataArray>(v.values[5]);
  EXPECT_THAT(pool_array->kind, Eq(MetadataKind::kMetadata));
  EXPECT_THAT(pool_array->type_key, StrEq("metadata.TensorInfoNode"));
  auto pool1 = Downcast<TensorInfo>(pool_array->array[0]);

  EXPECT_THAT(pool1->name(), Eq("pool1"));

  auto num_pools = Downcast<tvm::IntImm>(v.values[6]);
  EXPECT_THAT(num_pools->value, Eq(1));

  auto mod_name = Downcast<tvm::String>(v.values[7]);
  EXPECT_THAT(mod_name, Eq("default"));
}

using ::tvm::runtime::make_object;
TEST(Metadata, InMemory) {
  Metadata md = Metadata(make_object<tvm::target::metadata::InMemoryMetadataNode>(
      TVM_METADATA_VERSION,
      std::vector<TensorInfo>(
          {TensorInfo(make_object<tvm::target::metadata::InMemoryTensorInfoNode>(
               tvm::String("Input1"), std::vector<int64_t>{1, 5, 5, 3},
               tvm::runtime::DataType(DLDataType{1, 2, 3}))),
           TensorInfo(make_object<tvm::target::metadata::InMemoryTensorInfoNode>(
               tvm::String("Input2"), std::vector<int64_t>{1, 5, 5, 3},
               tvm::runtime::DataType(DLDataType{2, 3, 4})))}),
      std::vector<TensorInfo>(
          {TensorInfo(make_object<tvm::target::metadata::InMemoryTensorInfoNode>(
              tvm::String("Output1"), std::vector<int64_t>{3, 8, 8},
              tvm::runtime::DataType(DLDataType{3, 4, 5})))}),
      std::vector<TensorInfo>(
          {TensorInfo(make_object<tvm::target::metadata::InMemoryTensorInfoNode>(
              tvm::String("Pool1"), std::vector<int64_t>{5, 10, 10},
              tvm::runtime::DataType(DLDataType{3, 4, 7})))}),
      "default"));

  auto md_data = md->data();
  EXPECT_THAT(md_data->version, Eq(TVM_METADATA_VERSION));
  EXPECT_THAT(md_data->num_inputs, Eq(2));

  auto input0 = &md_data->inputs[0];
  EXPECT_THAT(input0->name, StrEq("Input1"));
  EXPECT_THAT(std::vector<int64_t>(input0->shape, input0->shape + input0->num_shape),
              ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(tvm::runtime::DataType(input0->dtype),
              Eq(tvm::runtime::DataType(DLDataType({1, 2, 3}))));

  auto input1 = &md_data->inputs[1];
  EXPECT_THAT(input1->name, StrEq("Input2"));
  EXPECT_THAT(std::vector<int64_t>(input1->shape, input1->shape + input1->num_shape),
              ElementsAre(1, 5, 5, 3));
  EXPECT_THAT(tvm::runtime::DataType(input1->dtype),
              Eq(tvm::runtime::DataType(DLDataType({2, 3, 4}))));

  auto output0 = &md_data->outputs[0];
  EXPECT_THAT(output0->name, StrEq("Output1"));
  EXPECT_THAT(std::vector<int64_t>(output0->shape, output0->shape + output0->num_shape),
              ElementsAre(3, 8, 8));
  EXPECT_THAT(tvm::runtime::DataType(output0->dtype),
              Eq(tvm::runtime::DataType(DLDataType({3, 4, 5}))));

  auto pool0 = &md_data->pools[0];
  EXPECT_THAT(pool0->name, StrEq("Pool1"));
  EXPECT_THAT(std::vector<int64_t>(pool0->shape, pool0->shape + pool0->num_shape),
              ElementsAre(5, 10, 10));
  EXPECT_THAT(tvm::runtime::DataType(pool0->dtype),
              Eq(tvm::runtime::DataType(DLDataType({3, 4, 7}))));

  EXPECT_THAT(md_data->mod_name, StrEq("default"));
}

TEST(Metadata, ZeroElementLists) {
  Metadata md = Metadata(make_object<tvm::target::metadata::InMemoryMetadataNode>(
      TVM_METADATA_VERSION, std::vector<TensorInfo>({}),
      std::vector<TensorInfo>(
          {TensorInfo(make_object<tvm::target::metadata::InMemoryTensorInfoNode>(
              tvm::String("Output1"), std::vector<int64_t>{},
              tvm::runtime::DataType(DLDataType{3, 4, 5})))}),
      std::vector<TensorInfo>({}), "default"));

  EXPECT_THAT(md->data()->num_inputs, Eq(0));
  EXPECT_THAT(md->inputs().size(), Eq(0));
  EXPECT_THAT(md->num_inputs(), Eq(0));
  EXPECT_THAT(md->inputs(), ElementsAre());

  auto output0 = md->data()->outputs[0];
  EXPECT_THAT(output0.num_shape, Eq(0));
  EXPECT_THAT(md->outputs()[0]->shape().size(), Eq(0));
  EXPECT_THAT(md->outputs()[0]->shape(), ElementsAre());

  EXPECT_THAT(md->pools().size(), Eq(0));
  EXPECT_THAT(md->num_pools(), Eq(0));
  EXPECT_THAT(md->pools(), ElementsAre());
}

TEST(MetadataArray, GetElementCStructName) {
  MetadataArray arr_struct{make_object<tvm::runtime::metadata::MetadataArrayNode>(
      Array<ObjectRef>(), MetadataKind::kMetadata, "metadata.FooMetadataNode")};
  EXPECT_THAT(arr_struct->kind, Eq(MetadataKind::kMetadata));
  EXPECT_THAT(arr_struct->get_element_c_struct_name(), StrEq("TVMFooMetadata"));

  MetadataArray arr_int{make_object<tvm::runtime::metadata::MetadataArrayNode>(
      Array<ObjectRef>(), MetadataKind::kInt64, nullptr)};
  EXPECT_THROW(arr_int->get_element_c_struct_name(), std::runtime_error);
}

namespace {
std::string ExplainDiscoveredNameEq(bool negation, std::string expected_name) {
  std::stringstream ss;
  ss << "std::get<0>(discovered_array) " << (negation ? "isn't" : "is") << " equal to "
     << expected_name;
  return ss.str();
}
}  // namespace

MATCHER_P(DiscoveredNameEq, expected_name, ExplainDiscoveredNameEq(negation, expected_name)) {
  return std::string(std::get<0>(arg)) == expected_name;
}

TEST(DiscoverArraysVisitor, DiscoverArrays) {
  std::vector<DiscoverArraysVisitor::DiscoveredArray> q;
  DiscoverArraysVisitor visitor(&q);

  Metadata md = Metadata(&kNormal);
  visitor.Visit(kMetadataGlobalSymbol, &md);

  EXPECT_THAT(q, ElementsAreArray({DiscoveredNameEq("kTvmgenMetadata_inputs_0_shape"),
                                   DiscoveredNameEq("kTvmgenMetadata_inputs_1_shape"),
                                   DiscoveredNameEq("kTvmgenMetadata_inputs"),
                                   DiscoveredNameEq("kTvmgenMetadata_outputs_0_shape"),
                                   DiscoveredNameEq("kTvmgenMetadata_outputs"),
                                   DiscoveredNameEq("kTvmgenMetadata_pools_0_shape"),
                                   DiscoveredNameEq("kTvmgenMetadata_pools")}));
}

// In Debug builds the _type_key is no longer inlined but also has no
// link-time definition.
#define WITH_TYPE_KEY 0

template <typename T,
          std::enable_if_t<std::is_base_of<tvm::runtime::metadata::MetadataBase, T>::value, bool> =
              true>
class TVMObjectIsInstanceMatcher : public MatcherInterface<tvm::runtime::metadata::MetadataBase> {
 public:
  using is_gtest_matcher = void;

  bool MatchAndExplain(tvm::runtime::metadata::MetadataBase arg,
                       MatchResultListener* os) const override {
    bool result = arg->IsInstance<typename T::ContainerType>();
    if (!result) {
#if WITH_TYPE_KEY
      (*os) << "is an instance of type " << T::ContainerType::_type_key;
#else
      (*os) << "is not of expected instance type";
#endif
    }

    return result;
  }

  void DescribeTo(std::ostream* os) const override {
#if WITH_TYPE_KEY
    (*os) << "is an instance of type " << T::ContainerType::_type_key;
#else
    (*os) << "is not of expected instance type";
#endif
  }

  void DescribeNegationTo(std::ostream* os) const override {
#if WITH_TYPE_KEY
    (*os) << "is not an instance of type " << T::ContainerType::_type_key;
#else
    (*os) << "is not of expected instance type";
#endif
  }
};

template <typename T>
Matcher<tvm::runtime::metadata::MetadataBase> TVMObjectIsInstance() {
  return Matcher<tvm::runtime::metadata::MetadataBase>(new TVMObjectIsInstanceMatcher<T>());
}

TEST(DiscoverComplexTypesVisitor, DiscoverComplexTypes) {
  std::vector<tvm::runtime::metadata::MetadataBase> q;
  DiscoverComplexTypesVisitor visitor(&q);

  Metadata md = Metadata(&kNormal);
  visitor.Discover(md);

  EXPECT_THAT(q, ElementsAre(TVMObjectIsInstance<TensorInfo>(), TVMObjectIsInstance<Metadata>()));
}
