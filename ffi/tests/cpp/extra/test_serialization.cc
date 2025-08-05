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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/string.h>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Serialization, BoolNull) {
  json::Object expected_null =
      json::Object{{"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "None"}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(nullptr), expected_null));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_null), nullptr));

  json::Object expected_true = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "bool"}, {"data", true}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(true), expected_true));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_true), true));

  json::Object expected_false = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "bool"}, {"data", false}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(false), expected_false));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_false), false));
}

TEST(Serialization, IntegerTypes) {
  // Test positive integer
  json::Object expected_int = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "int"}, {"data", 42}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(static_cast<int64_t>(42)), expected_int));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_int), static_cast<int64_t>(42)));
}

TEST(Serialization, FloatTypes) {
  // Test positive float
  json::Object expected_float =
      json::Object{{"root_index", 0},
                   {"nodes", json::Array{json::Object{{"type", "float"}, {"data", 3.14159}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(3.14159), expected_float));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_float), 3.14159));
}

TEST(Serialization, StringTypes) {
  // Test short string
  json::Object expected_short = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"}, {"data", String("hello")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(String("hello")), expected_short));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_short), String("hello")));

  // Test long string
  std::string long_str(1000, 'x');
  json::Object expected_long = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"}, {"data", String(long_str)}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(String(long_str)), expected_long));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_long), String(long_str)));

  // Test string with special characters
  json::Object expected_special = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"},
                                         {"data", String("hello\nworld\t\"quotes\"")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(String("hello\nworld\t\"quotes\"")), expected_special));
  EXPECT_TRUE(
      StructuralEqual()(FromJSONGraph(expected_special), String("hello\nworld\t\"quotes\"")));
}

TEST(Serialization, Bytes) {
  // Test empty bytes
  Bytes empty_bytes;
  json::Object expected_empty = json::Object{
      {"root_index", 0}, {"nodes", json::Array{json::Object{{"type", "ffi.Bytes"}, {"data", ""}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_bytes), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_bytes));

  // Test bytes with that encoded as base64
  Bytes bytes_content = Bytes("abcd");
  json::Object expected_encoded = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Bytes"}, {"data", "YWJjZA=="}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(bytes_content), expected_encoded));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_encoded), bytes_content));

  // Test bytes with that encoded as base64, that contains control characters via utf-8
  char bytes_v2_content[] = {0x01, 0x02, 0x03, 0x04, 0x01, 0x0b};
  Bytes bytes_v2 = Bytes(bytes_v2_content, sizeof(bytes_v2_content));
  json::Object expected_encoded_v2 = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Bytes"}, {"data", "AQIDBAEL"}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(bytes_v2), expected_encoded_v2));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_encoded_v2), bytes_v2));
}

TEST(Serialization, DataTypes) {
  // Test int32 dtype
  DLDataType int32_dtype;
  int32_dtype.code = kDLInt;
  int32_dtype.bits = 32;
  int32_dtype.lanes = 1;

  json::Object expected_int32 = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "DataType"}, {"data", String("int32")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(int32_dtype), expected_int32));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_int32), int32_dtype));

  // Test float64 dtype
  DLDataType float64_dtype;
  float64_dtype.code = kDLFloat;
  float64_dtype.bits = 64;
  float64_dtype.lanes = 1;

  json::Object expected_float64 = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "DataType"}, {"data", String("float64")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(float64_dtype), expected_float64));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_float64), float64_dtype));

  // Test vector dtype
  DLDataType vector_dtype;
  vector_dtype.code = kDLFloat;
  vector_dtype.bits = 32;
  vector_dtype.lanes = 4;

  json::Object expected_vector = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "DataType"}, {"data", String("float32x4")}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(vector_dtype), expected_vector));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_vector), vector_dtype));
}

TEST(Serialization, DeviceTypes) {
  // Test CPU device
  DLDevice cpu_device;
  cpu_device.device_type = kDLCPU;
  cpu_device.device_id = 0;

  json::Object expected_cpu = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "Device"},
                                         {"data", json::Array{static_cast<int64_t>(kDLCPU),
                                                              static_cast<int64_t>(0)}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(cpu_device), expected_cpu));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_cpu), cpu_device));

  // Test GPU device
  DLDevice gpu_device;
  gpu_device.device_type = kDLCUDA;
  gpu_device.device_id = 1;

  json::Object expected_gpu = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{
                    {"type", "Device"}, {"data", json::Array{static_cast<int64_t>(kDLCUDA), 1}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(gpu_device), expected_gpu));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_gpu), gpu_device));
}

TEST(Serialization, Arrays) {
  // Test empty array
  Array<Any> empty_array;
  json::Object expected_empty = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Array"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_array), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_array));

  // Test single element array
  Array<Any> single_array;
  single_array.push_back(Any(42));
  json::Object expected_single =
      json::Object{{"root_index", 1},
                   {"nodes", json::Array{
                                 json::Object{{"type", "int"}, {"data", static_cast<int64_t>(42)}},
                                 json::Object{{"type", "ffi.Array"}, {"data", json::Array{0}}},
                             }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(single_array), expected_single));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_single), single_array));

  // Test duplicated element array
  Array<Any> duplicated_array;
  duplicated_array.push_back(42);
  duplicated_array.push_back(42);
  json::Object expected_duplicated =
      json::Object{{"root_index", 1},
                   {"nodes", json::Array{
                                 json::Object{{"type", "int"}, {"data", 42}},
                                 json::Object{{"type", "ffi.Array"}, {"data", json::Array{0, 0}}},
                             }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(duplicated_array), expected_duplicated));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_duplicated), duplicated_array));
  // Test mixed element array, note that 42 and "hello" are duplicated and will
  // be indexed as 0 and 1
  Array<Any> mixed_array;
  mixed_array.push_back(42);
  mixed_array.push_back(String("hello"));
  mixed_array.push_back(true);
  mixed_array.push_back(nullptr);
  mixed_array.push_back(42);
  mixed_array.push_back(String("hello"));
  json::Object expected_mixed = json::Object{
      {"root_index", 4},
      {"nodes", json::Array{
                    json::Object{{"type", "int"}, {"data", 42}},
                    json::Object{{"type", "ffi.String"}, {"data", String("hello")}},
                    json::Object{{"type", "bool"}, {"data", true}},
                    json::Object{{"type", "None"}},
                    json::Object{{"type", "ffi.Array"}, {"data", json::Array{0, 1, 2, 3, 0, 1}}},
                }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(mixed_array), expected_mixed));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_mixed), mixed_array));
}

TEST(Serialization, Maps) {
  // Test empty map
  Map<String, Any> empty_map;
  json::Object expected_empty = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Map"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_map), expected_empty));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty), empty_map));

  // Test single element map
  Map<String, Any> single_map{{"key", 42}};
  json::Object expected_single = json::Object{
      {"root_index", 2},
      {"nodes", json::Array{json::Object{{"type", "ffi.String"}, {"data", String("key")}},
                            json::Object{{"type", "int"}, {"data", 42}},
                            json::Object{{"type", "ffi.Map"}, {"data", json::Array{0, 1}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(single_map), expected_single));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_single), single_map));

  // Test duplicated element map
  Map<String, Any> duplicated_map{{"b", 42}, {"a", 42}};
  json::Object expected_duplicated = json::Object{
      {"root_index", 3},
      {"nodes", json::Array{
                    json::Object{{"type", "ffi.String"}, {"data", "b"}},
                    json::Object{{"type", "int"}, {"data", 42}},
                    json::Object{{"type", "ffi.String"}, {"data", "a"}},
                    json::Object{{"type", "ffi.Map"}, {"data", json::Array{0, 1, 2, 1}}},

                }}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(duplicated_map), expected_duplicated));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_duplicated), duplicated_map));
}

TEST(Serialization, Shapes) {
  Shape empty_shape;

  json::Object expected_empty_shape = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Shape"}, {"data", json::Array{}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(empty_shape), expected_empty_shape));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_empty_shape), empty_shape));

  Shape shape({1, 2, 3});
  json::Object expected_shape = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{json::Object{{"type", "ffi.Shape"}, {"data", json::Array{1, 2, 3}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(shape), expected_shape));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_shape), shape));
}

TEST(Serialization, TestObjectVar) {
  TVar x = TVar("x");
  json::Object expected_x = json::Object{
      {"root_index", 1},
      {"nodes",
       json::Array{json::Object{{"type", "ffi.String"}, {"data", "x"}},
                   json::Object{{"type", "test.Var"}, {"data", json::Object{{"name", 0}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(x), expected_x));
  EXPECT_TRUE(StructuralEqual::Equal(FromJSONGraph(expected_x), x, /*map_free_vars=*/true));
}

TEST(Serialization, TestObjectIntCustomToJSON) {
  TInt value = TInt(42);
  json::Object expected_i = json::Object{
      {"root_index", 0},
      {"nodes",
       json::Array{json::Object{{"type", "test.Int"}, {"data", json::Object{{"value", 42}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(value), expected_i));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_i), value));
}

TEST(Serialization, TestObjectFunc) {
  TVar x = TVar("x");
  // comment fields are ignored
  TFunc fa = TFunc({x}, {x, x}, String("comment a"));

  json::Object expected_fa = json::Object{
      {"root_index", 5},
      {"nodes",
       json::Array{
           json::Object{{"type", "ffi.String"}, {"data", "x"}},                      // string "x"
           json::Object{{"type", "test.Var"}, {"data", json::Object{{"name", 0}}}},  // var x
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{1}}},            // array [x]
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{1, 1}}},         // array [x, x]
           json::Object{{"type", "ffi.String"}, {"data", "comment a"}},              // "comment a"
           json::Object{{"type", "test.Func"},
                        {"data", json::Object{{"params", 2}, {"body", 3}, {"comment", 4}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(fa), expected_fa));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_fa), fa));

  TFunc fb = TFunc({}, {}, std::nullopt);
  json::Object expected_fb = json::Object{
      {"root_index", 3},
      {"nodes",
       json::Array{
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{}}},
           json::Object{{"type", "ffi.Array"}, {"data", json::Array{}}},
           json::Object{{"type", "None"}},
           json::Object{{"type", "test.Func"},
                        {"data", json::Object{{"params", 0}, {"body", 1}, {"comment", 2}}}}}}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(fb), expected_fb));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_fb), fb));
}

TEST(Serialization, AttachMetadata) {
  bool value = true;
  json::Object metadata{{"version", "1.0"}};
  json::Object expected =
      json::Object{{"root_index", 0},
                   {"nodes", json::Array{json::Object{{"type", "bool"}, {"data", true}}}},
                   {"metadata", metadata}};
  EXPECT_TRUE(StructuralEqual()(ToJSONGraph(value, metadata), expected));
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected), value));
}

TEST(Serialization, ShuffleNodeOrder) {
  // the FromJSONGraph is agnostic to the node order
  // so we can shuffle the node order as it reads nodes lazily
  Map<String, Any> duplicated_map{{"b", 42}, {"a", 42}};
  json::Object expected_shuffled = json::Object{
      {"root_index", 0},
      {"nodes", json::Array{
                    json::Object{{"type", "ffi.Map"}, {"data", json::Array{2, 3, 1, 3}}},
                    json::Object{{"type", "ffi.String"}, {"data", "a"}},
                    json::Object{{"type", "ffi.String"}, {"data", "b"}},
                    json::Object{{"type", "int"}, {"data", 42}},
                }}};
  EXPECT_TRUE(StructuralEqual()(FromJSONGraph(expected_shuffled), duplicated_map));
}

}  // namespace
