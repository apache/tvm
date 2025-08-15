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
#include <tvm/ffi/extra/structural_equal.h>

#include <cmath>

namespace {

using namespace tvm::ffi;

inline bool FastMathSafeIsNaN(double x) {
#ifdef __FAST_MATH__
  // Bit-level NaN detection (IEEE 754 double)
  // IEEE 754 standard: https://en.wikipedia.org/wiki/IEEE_754
  // NaN is encoded as all 1s in the exponent and non-zero in the mantissa
  static_assert(sizeof(double) == sizeof(uint64_t), "Unexpected double size");
  uint64_t bits = *reinterpret_cast<const uint64_t*>(&x);
  uint64_t exponent = (bits >> 52) & 0x7FF;
  uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;
  return (exponent == 0x7FF) && (mantissa != 0);
#else
  // Safe to use std::isnan when fast-math is off
  return std::isnan(x);
#endif
}

inline bool FastMathSafeIsInf(double x) {
#ifdef __FAST_MATH__
  // IEEE 754 standard: https://en.wikipedia.org/wiki/IEEE_754
  // Inf is encoded as all 1s in the exponent and zero in the mantissa
  static_assert(sizeof(double) == sizeof(uint64_t), "Unexpected double size");
  uint64_t bits = *reinterpret_cast<const uint64_t*>(&x);
  uint64_t exponent = (bits >> 52) & 0x7FF;
  uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;
  // inf is encoded as all 1s in the exponent and zero in the mantissa
  return (exponent == 0x7FF) && (mantissa == 0);
#else
  return std::isinf(x);
#endif
}

TEST(JSONParser, BoolNull) {
  // boolean value
  EXPECT_EQ(json::Parse("true").cast<bool>(), true);
  EXPECT_EQ(json::Parse("false").cast<bool>(), false);
  EXPECT_EQ(json::Parse("null"), nullptr);
}

TEST(JSONParser, WrongBoolNull) {
  String error_msg;
  EXPECT_EQ(json::Parse("nul", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
  EXPECT_EQ(json::Parse("fals", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
  EXPECT_EQ(json::Parse("\n\nfx", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 3 column 1 (char 2)");
  EXPECT_EQ(json::Parse("fx", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
  EXPECT_EQ(json::Parse("n1", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
  EXPECT_EQ(json::Parse("t1", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
  EXPECT_EQ(json::Parse("f1", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
}

TEST(JSONParser, Number) {
  // number
  EXPECT_EQ(json::Parse("123").cast<int64_t>(), 123);
  EXPECT_EQ(json::Parse("-124").cast<int64_t>(), -124);
  EXPECT_EQ(json::Parse("123.456").cast<double>(), 123.456);
  // parsing scientific notation
  EXPECT_EQ(json::Parse("1.456e12").cast<double>(), 1.456e12);
  // NaN
  EXPECT_EQ(FastMathSafeIsNaN(json::Parse("NaN").cast<double>()), true);
  // Infinity
  EXPECT_EQ(FastMathSafeIsInf(json::Parse("Infinity").cast<double>()), true);
  // -Infinity
  EXPECT_EQ(FastMathSafeIsInf(-json::Parse("-Infinity").cast<double>()), true);

  // Test zero variants
  EXPECT_EQ(json::Parse("0").cast<int64_t>(), 0);
  EXPECT_EQ(json::Parse("-0").cast<double>(), -0.0);
  EXPECT_EQ(json::Parse("0.0").cast<double>(), 0.0);

  // Test very large numbers
  EXPECT_EQ(json::Parse("9223372036854775807").cast<int64_t>(),
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(json::Parse("-9223372036854775808").cast<int64_t>(),
            std::numeric_limits<int64_t>::min());

  // Test very small decimals
  EXPECT_EQ(json::Parse("1e-10").cast<double>(), 1e-10);
  EXPECT_EQ(json::Parse("-1e-10").cast<double>(), -1e-10);

  // Test scientific notation edge cases
  EXPECT_EQ(json::Parse("1E+10").cast<double>(), 1E+10);
  EXPECT_EQ(json::Parse("1e+10").cast<double>(), 1e+10);
  EXPECT_EQ(json::Parse("1E-10").cast<double>(), 1E-10);
  EXPECT_EQ(json::Parse("123.456E+10").cast<double>(), 123.456E+10);
}

TEST(JSONParser, WrongNumber) {
  String error_msg;
  EXPECT_EQ(json::Parse("123.456.789", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");

  // Test invalid number formats
  EXPECT_EQ(json::Parse("123e", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
  EXPECT_EQ(json::Parse("123e+", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
  EXPECT_EQ(json::Parse("123E-", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");
}

TEST(JSONParser, String) {
  EXPECT_EQ(json::Parse("\"hello\"").cast<String>(), "hello");
  EXPECT_EQ(json::Parse("\n\t \"hello\"\n\r").cast<String>(), "hello");
  EXPECT_EQ(json::Parse("\"hello\\nworld\"").cast<String>(), "hello\nworld");
  EXPECT_EQ(json::Parse("\"\"").cast<String>(), "");
  // test escape characters
  EXPECT_EQ(json::Parse("\"\\ta\\n\\/\\f\\\"\\\\\"").cast<String>(), "\ta\n/\f\"\\");
  // test unicode code point
  EXPECT_EQ(json::Parse("\"\\u0041\"").cast<String>(), "A");
  // test unicode surrogate pair
  EXPECT_EQ(json::Parse("\"\\uD83D\\uDE04hello\"").cast<String>(), u8"\U0001F604hello");
}

TEST(JSONParser, WrongString) {
  String error_msg;
  EXPECT_EQ(json::Parse("\"hello", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Unterminated string starting at: line 1 column 1 (char 0)");

  EXPECT_EQ(json::Parse("\"hello\x01\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Invalid control character at: line 1 column 7 (char 6)");

  EXPECT_EQ(json::Parse("\"hello\\uxx\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Invalid \\uXXXX escape: line 1 column 8 (char 7)");

  EXPECT_EQ(json::Parse("\"hello\\uDC00\\uDE04\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Invalid surrogate pair of \\uXXXX escapes: line 1 column 8 (char 7)");

  EXPECT_EQ(json::Parse("\"hello\\uD800\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Invalid surrogate pair of \\uXXXX escapes: line 1 column 8 (char 7)");

  EXPECT_EQ(json::Parse("\"hello\\uD800\\uxx\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Invalid \\uXXXX escape: line 1 column 15 (char 14)");

  EXPECT_EQ(json::Parse("\"hello\\a\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Invalid \\escape: line 1 column 8 (char 7)");
}

TEST(JSONParser, Array) {
  EXPECT_TRUE(StructuralEqual()(json::Parse("[]"), json::Array{}));

  EXPECT_TRUE(StructuralEqual()(json::Parse("[1, 2,\n\t\"a\"]"), json::Array{1, 2, "a"}));
}

TEST(JSONParser, WrongArray) {
  String error_msg;

  EXPECT_EQ(json::Parse("]", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");

  EXPECT_EQ(json::Parse("[1,]", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 4 (char 3)");

  EXPECT_EQ(json::Parse("[", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 2 (char 1)");

  EXPECT_EQ(json::Parse("[1a", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting ',' delimiter: line 1 column 3 (char 2)");

  EXPECT_EQ(json::Parse("[1,2,3", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting ',' delimiter: line 1 column 7 (char 6)");

  EXPECT_EQ(json::Parse("[1]  a", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Extra data: line 1 column 6 (char 5)");
}

TEST(JSONParser, Object) {
  EXPECT_TRUE(StructuralEqual()(json::Parse("{}"), json::Object{}));

  EXPECT_TRUE(StructuralEqual()(json::Parse("{\"a\":  1, \n\"b\": \t\"c\"}   "),
                                json::Object{{"a", 1}, {"b", "c"}}));
}

TEST(JSONParser, ObjectOrderPreserving) {
  auto obj = json::Parse("{\"c\": 1, \"a\": 2, \"b\": 3}   ");
  json::Array keys;
  for (auto& [key, value] : obj.cast<json::Object>()) {
    keys.push_back(key);
  }
  EXPECT_TRUE(StructuralEqual()(keys, json::Array{"c", "a", "b"}));
}

TEST(JSONParser, WrongObject) {
  String error_msg;
  EXPECT_EQ(json::Parse("{\"a\":", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 6 (char 5)");

  EXPECT_EQ(json::Parse("{", &error_msg), nullptr);
  EXPECT_EQ(error_msg,
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)");

  // Test incomplete structures
  EXPECT_EQ(json::Parse("{\"incomplete\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting ':' delimiter: line 1 column 14 (char 13)");
}

TEST(JSONParser, NestedObject) {
  EXPECT_TRUE(
      StructuralEqual()(json::Parse("{\"a\": \t{\"b\": 1}, \n\"c\": [1, 2, 3]}"),
                        json::Object{{"a", json::Object{{"b", 1}}}, {"c", json::Array{1, 2, 3}}}));

  EXPECT_TRUE(StructuralEqual()(
      json::Parse("{\"a\": \t{\"b\": 1}, \n\"c\": [1, null, Infinity]}"),
      json::Object{{"a", json::Object{{"b", 1}}},
                   {"c", json::Array{1, nullptr, std::numeric_limits<double>::infinity()}}}));

  EXPECT_TRUE(StructuralEqual()(
      json::Parse("[{}, {\"a\": [1.1, 1000000]}]"),
      json::Array{json::Object{}, json::Object{{"a", json::Array{1.1, 1000000}}}}));
}

TEST(JSONParser, WrongNestedObject) {
  String error_msg;
  EXPECT_EQ(json::Parse("{\"a\":\n\n[1]", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting ',' delimiter: line 3 column 4 (char 10)");

  EXPECT_EQ(json::Parse("{\"a\":\n\n[abc]}", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 3 column 2 (char 8)");
}

// edge cases
TEST(JSONParser, WhitespaceHandling) {
  // Test various whitespace characters
  EXPECT_EQ(json::Parse(" \t\n\r true \t\n\r ").cast<bool>(), true);
  EXPECT_EQ(json::Parse("\n\n\n123\n\n\n").cast<int64_t>(), 123);
  EXPECT_EQ(json::Parse("   \"hello world\"   ").cast<String>(), "hello world");

  // Test whitespace in arrays and objects
  EXPECT_TRUE(StructuralEqual()(json::Parse("  [  1  ,  2  ,  3  ]  "), json::Array{1, 2, 3}));

  EXPECT_TRUE(StructuralEqual()(json::Parse("  {  \"a\"  :  1  ,  \"b\"  :  2  }  "),
                                json::Object{{"a", 1}, {"b", 2}}));
}

TEST(JSONParser, WrongEmptyAndMinimalInputs) {
  String error_msg;
  // Test empty string
  EXPECT_EQ(json::Parse("", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 1 column 1 (char 0)");

  // Test only whitespace
  EXPECT_EQ(json::Parse("   \t\n    ", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Expecting value: line 2 column 5 (char 9)");
}

TEST(JSONParser, UnicodeEdgeCases) {
  // Test various unicode characters
  EXPECT_EQ(json::Parse("\"\\u0000\"").cast<String>(), std::string("\0", 1));
  // replace using \U to avoid encoding issues
  EXPECT_EQ(json::Parse("\"\\u00FF\"").cast<String>(), u8"\U000000FF");
  EXPECT_EQ(json::Parse("\"\\u4E2D\\u6587\"").cast<String>(), u8"\U00004E2D\U00006587");

  // Test multiple surrogate pairs
  EXPECT_EQ(json::Parse("\"\\uD83D\\uDE00\\uD83D\\uDE01\"").cast<String>(),
            u8"\U0001F600\U0001F601");
}

TEST(JSONParser, LargeInputs) {
  // Test large array
  std::string large_array = "[";
  for (int i = 0; i < 1000; ++i) {
    if (i > 0) large_array += ",";
    large_array += std::to_string(i);
  }
  large_array += "]";

  auto result = json::Parse(large_array);
  EXPECT_TRUE(result != nullptr);
  EXPECT_EQ(result.cast<json::Array>().size(), 1000);

  // Test large object
  std::string large_object = "{";
  for (int i = 0; i < 500; ++i) {
    if (i > 0) large_object += ",";
    large_object += "\"key" + std::to_string(i) + "\":" + std::to_string(i);
  }
  large_object += "}";

  result = json::Parse(large_object);
  EXPECT_TRUE(result != nullptr);
  EXPECT_EQ(result.cast<json::Object>().size(), 500);
}

TEST(JSONParser, MixedDataTypes) {
  // Test complex nested structure with all data types
  std::string complex_json = R"({
    "null_value": null,
    "boolean_true": true,
    "boolean_false": false,
    "integer": 42,
    "negative_integer": -42,
    "float": 3.14159,
    "scientific": 1.23e-4,
    "string": "hello world",
    "unicode_string": "Hello \u4e16\u754c \ud83c\udf0d",
    "empty_string": "",
    "empty_array": [],
    "empty_object": {},
    "number_array": [1, 2, 3, 4, 5],
    "mixed_array": [1, "two", true, null, 3.14],
    "nested_object": {
      "level1": {
        "level2": {
          "data": [1, 2, {"nested_array": [true, false]}]
        }
      }
    }
  })";

  auto result = json::Parse(complex_json);

  // Create expected structure for comparison
  json::Object expected{
      {"null_value", nullptr},
      {"boolean_true", true},
      {"boolean_false", false},
      {"integer", 42},
      {"negative_integer", -42},
      {"float", 3.14159},
      {"scientific", 1.23e-4},
      {"string", "hello world"},
      {"unicode_string", u8"Hello \U00004E16\U0000754C \U0001F30D"},
      {"empty_string", ""},
      {"empty_array", json::Array{}},
      {"empty_object", json::Object{}},
      {"number_array", json::Array{1, 2, 3, 4, 5}},
      {"mixed_array", json::Array{1, "two", true, nullptr, 3.14}},
      {"nested_object",
       json::Object{
           {"level1",
            json::Object{
                {"level2",
                 json::Object{
                     {"data",
                      json::Array{1, 2,
                                  json::Object{{"nested_array", json::Array{true, false}}}}}}}}}}}};

  EXPECT_TRUE(StructuralEqual()(result, expected));
}

TEST(JSONParser, WrongExtraData) {
  String error_msg;

  EXPECT_EQ(json::Parse("truee", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Extra data: line 1 column 5 (char 4)");

  EXPECT_EQ(json::Parse("true false", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Extra data: line 1 column 6 (char 5)");

  EXPECT_EQ(json::Parse("123 456", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Extra data: line 1 column 5 (char 4)");

  EXPECT_EQ(json::Parse("\"hello\" \"world\"", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Extra data: line 1 column 9 (char 8)");

  EXPECT_EQ(json::Parse("{} []", &error_msg), nullptr);
  EXPECT_EQ(error_msg, "Extra data: line 1 column 4 (char 3)");
}
}  // namespace
