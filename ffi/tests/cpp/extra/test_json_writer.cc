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
#include <tvm/ffi/extra/json.h>

#include <limits>

namespace {

using namespace tvm::ffi;

TEST(JSONWriter, BoolNull) {
  // boolean value
  EXPECT_EQ(json::Stringify(json::Value(true)), "true");
  EXPECT_EQ(json::Stringify(json::Value(false)), "false");
  EXPECT_EQ(json::Stringify(json::Value(nullptr)), "null");
}

TEST(JSONWriter, Integer) {
  // positive integer
  EXPECT_EQ(json::Stringify(json::Value(42)), "42");
  // negative integer
  EXPECT_EQ(json::Stringify(json::Value(-123)), "-123");
  // zero
  EXPECT_EQ(json::Stringify(json::Value(0)), "0");
  // large positive integer
  EXPECT_EQ(json::Stringify(json::Value(std::numeric_limits<int64_t>::max())),
            "9223372036854775807");
  // large negative integer
  EXPECT_EQ(json::Stringify(json::Value(std::numeric_limits<int64_t>::min())),
            "-9223372036854775808");
}

TEST(JSONWriter, Float) {
  // regular float
  EXPECT_EQ(json::Stringify(json::Value(2.5)), "2.5");
  // integer-like float (should have .0 suffix)
  EXPECT_EQ(json::Stringify(json::Value(5.0)), "5.0");
  EXPECT_EQ(json::Stringify(json::Value(-10.0)), "-10.0");
  // zero float
  EXPECT_EQ(json::Stringify(json::Value(0.0)), "0.0");
  // scientific notation for very small numbers
  EXPECT_EQ(json::Stringify(json::Value(-7.89e-15)), "-7.89e-15");
  // short scientific notation (shorter than fixed-point)
  EXPECT_EQ(json::Stringify(json::Value(2e-8)), "2e-08");
  // NaN
  EXPECT_EQ(json::Stringify(json::Value(std::numeric_limits<double>::quiet_NaN())), "NaN");
  // positive infinity
  EXPECT_EQ(json::Stringify(json::Value(std::numeric_limits<double>::infinity())), "Infinity");
  // negative infinity
  EXPECT_EQ(json::Stringify(json::Value(-std::numeric_limits<double>::infinity())), "-Infinity");
}

TEST(JSONWriter, String) {
  // simple string
  EXPECT_EQ(json::Stringify(json::Value(String("hello"))), "\"hello\"");
  // empty string
  EXPECT_EQ(json::Stringify(json::Value(String(""))), "\"\"");
  // string with escaped characters
  EXPECT_EQ(json::Stringify(json::Value(String("\"quoted\""))), "\"\\\"quoted\\\"\"");
  EXPECT_EQ(json::Stringify(json::Value(String("backslash\\"))), "\"backslash\\\\\"");
  EXPECT_EQ(json::Stringify(json::Value(String("forward/slash"))), "\"forward\\/slash\"");
  EXPECT_EQ(json::Stringify(json::Value(String("line\nbreak"))), "\"line\\nbreak\"");
  EXPECT_EQ(json::Stringify(json::Value(String("tab\there"))), "\"tab\\there\"");
  EXPECT_EQ(json::Stringify(json::Value(String("carriage\rreturn"))), "\"carriage\\rreturn\"");
  // string with control character
  EXPECT_EQ(json::Stringify(json::Value(String(std::string("\x01", 1) + "control"))),
            "\"\\u0001control\"");
}

TEST(JSONWriter, Array) {
  // empty array
  json::Array empty_array;
  EXPECT_EQ(json::Stringify(empty_array), "[]");

  // single element array
  json::Array single_array{42};
  EXPECT_EQ(json::Stringify(single_array), "[42]");

  // multiple elements array
  json::Array multi_array{1, "hello", true};
  EXPECT_EQ(json::Stringify(multi_array), "[1,\"hello\",true]");

  // nested array
  json::Array nested_array{json::Array{1, 2}, 3};
  EXPECT_EQ(json::Stringify(nested_array), "[[1,2],3]");
}

TEST(JSONWriter, Object) {
  // empty object
  json::Object empty_object;
  EXPECT_EQ(json::Stringify(empty_object), "{}");

  // single key-value pair
  json::Object single_object{{String("key"), String("value")}};
  EXPECT_EQ(json::Stringify(single_object), "{\"key\":\"value\"}");

  // multiple key-value pairs - insertion order preservation
  json::Object multi_object{{"name", "Alice"}, {"age", 30}, {"active", true}, {"score", 95.5}};
  EXPECT_EQ(json::Stringify(multi_object),
            "{\"name\":\"Alice\",\"age\":30,\"active\":true,\"score\":95.5}");
}

TEST(JSONWriter, InsertionOrderPreservation) {
  // test that objects preserve insertion order
  json::Object ordered_object{
      {"zebra", "last"}, {"alpha", "first"}, {"beta", "middle"}, {"gamma", 123}, {"delta", true}};
  EXPECT_EQ(
      json::Stringify(ordered_object),
      "{\"zebra\":\"last\",\"alpha\":\"first\",\"beta\":\"middle\",\"gamma\":123,\"delta\":true}");

  // test with indentation to verify order is preserved
  std::string ordered_indented = json::Stringify(ordered_object, 2);
  EXPECT_EQ(ordered_indented, String(R"({
  "zebra": "last",
  "alpha": "first",
  "beta": "middle",
  "gamma": 123,
  "delta": true
})"));

  // test nested objects also preserve order
  json::Object nested_ordered{
      {"outer1",
       json::Object{{"inner_z", "z_value"}, {"inner_a", "a_value"}, {"inner_m", "m_value"}}},
      {"outer2", json::Object{{"third", 3}, {"first", 1}, {"second", 2}}}};
  std::string nested_ordered_indented = json::Stringify(nested_ordered, 2);
  EXPECT_EQ(nested_ordered_indented, String(R"({
  "outer1": {
    "inner_z": "z_value",
    "inner_a": "a_value",
    "inner_m": "m_value"
  },
  "outer2": {
    "third": 3,
    "first": 1,
    "second": 2
  }
})"));
}

TEST(JSONWriter, NestedStructures) {
  // object containing array
  json::Object obj_with_array{{String("numbers"), json::Array{1, 2, 3}}};
  EXPECT_EQ(json::Stringify(obj_with_array), "{\"numbers\":[1,2,3]}");

  // array containing object
  json::Array arr_with_obj{json::Object{{String("key"), String("value")}}};
  EXPECT_EQ(json::Stringify(arr_with_obj), "[{\"key\":\"value\"}]");

  // deeply nested structure
  json::Object nested_obj{
      {String("nested"), json::Array{json::Object{{String("deep"), String("value")}}}}};
  EXPECT_EQ(json::Stringify(nested_obj), "{\"nested\":[{\"deep\":\"value\"}]}");
}

TEST(JSONWriter, Indentation) {
  // test with indentation
  json::Array arr{1, 2};
  std::string indented = json::Stringify(arr, 2);
  EXPECT_EQ(indented, String(R"([
  1,
  2
])"));

  // object with indentation
  json::Object obj{{"key", "value"}};
  std::string indented_obj = json::Stringify(obj, 2);
  EXPECT_EQ(indented_obj, String(R"({
  "key": "value"
})"));

  // complex nested structure with multiple data types
  // keep double as .5 so output is deterministic as they exactly rounds to power of 2
  json::Object complex_nested{
      {"name", "test"},
      {"count", 42},
      {"price", 3.5},
      {"active", true},
      {"metadata", nullptr},
      {"numbers", json::Array{1, 2, 3}},
      {"config", json::Object{{"enabled", false},
                              {"timeout", 30.5},
                              {"tags", json::Array{"production", "critical", nullptr}}}},
      {"matrix", json::Array{json::Array{1, 2}, json::Array{3.5, 4.5}, json::Array{"a", "b"}}}};
  std::string complex_indented = json::Stringify(complex_nested, 2);
  EXPECT_EQ(complex_indented, String(R"({
  "name": "test",
  "count": 42,
  "price": 3.5,
  "active": true,
  "metadata": null,
  "numbers": [
    1,
    2,
    3
  ],
  "config": {
    "enabled": false,
    "timeout": 30.5,
    "tags": [
      "production",
      "critical",
      null
    ]
  },
  "matrix": [
    [
      1,
      2
    ],
    [
      3.5,
      4.5
    ],
    [
      "a",
      "b"
    ]
  ]
})"));
}
}  // namespace
