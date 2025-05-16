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
#include <tvm/ffi/any.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/string.h>

namespace {

using namespace tvm::ffi;

TEST(String, MoveFromStd) {
  using namespace std;
  string source = "this is a string";
  string expect = source;
  String s(std::move(source));
  string copy = (string)s;
  EXPECT_EQ(copy, expect);
  EXPECT_EQ(source.size(), 0);
}

TEST(String, CopyFromStd) {
  using namespace std;
  string source = "this is a string";
  string expect = source;
  String s{source};
  string copy = (string)s;
  EXPECT_EQ(copy, expect);
  EXPECT_EQ(source.size(), expect.size());
}

TEST(String, Assignment) {
  using namespace std;
  String s{string{"hello"}};
  s = string{"world"};
  EXPECT_EQ(s == "world", true);
  string s2{"world2"};
  s = std::move(s2);
  EXPECT_EQ(s == "world2", true);

  ObjectRef r;
  r = String("hello");
  EXPECT_EQ(r.defined(), true);
}

TEST(String, empty) {
  using namespace std;
  String s{"hello"};
  EXPECT_EQ(s.empty(), false);
  s = std::string("");
  EXPECT_EQ(s.empty(), true);
}

TEST(String, Comparisons) {
  using namespace std;
  string source = "a string";
  string mismatch = "a string but longer";
  String s{"a string"};
  String m{mismatch};

  EXPECT_EQ("a str" >= s, false);
  EXPECT_EQ(s == source, true);
  EXPECT_EQ(s == mismatch, false);
  EXPECT_EQ(s == source.data(), true);
  EXPECT_EQ(s == mismatch.data(), false);

  EXPECT_EQ(s < m, source < mismatch);
  EXPECT_EQ(s > m, source > mismatch);
  EXPECT_EQ(s <= m, source <= mismatch);
  EXPECT_EQ(s >= m, source >= mismatch);
  EXPECT_EQ(s == m, source == mismatch);
  EXPECT_EQ(s != m, source != mismatch);

  EXPECT_EQ(m < s, mismatch < source);
  EXPECT_EQ(m > s, mismatch > source);
  EXPECT_EQ(m <= s, mismatch <= source);
  EXPECT_EQ(m >= s, mismatch >= source);
  EXPECT_EQ(m == s, mismatch == source);
  EXPECT_EQ(m != s, mismatch != source);
}

// Check '\0' handling
TEST(String, null_byte_handling) {
  using namespace std;
  // Ensure string still compares equal if it contains '\0'.
  string v1 = "hello world";
  size_t v1_size = v1.size();
  v1[5] = '\0';
  EXPECT_EQ(v1[5], '\0');
  EXPECT_EQ(v1.size(), v1_size);
  String str_v1{v1};
  EXPECT_EQ(str_v1.compare(v1), 0);
  EXPECT_EQ(str_v1.size(), v1_size);

  // Ensure bytes after '\0' are taken into account for mismatches.
  string v2 = "aaa one";
  string v3 = "aaa two";
  v2[3] = '\0';
  v3[3] = '\0';
  String str_v2{v2};
  String str_v3{v3};
  EXPECT_EQ(str_v2.compare(str_v3), -1);
  EXPECT_EQ(str_v2.size(), 7);
  // strcmp won't be able to detect the mismatch
  EXPECT_EQ(strcmp(v2.data(), v3.data()), 0);
  // string::compare can handle \0 since it knows size
  EXPECT_LT(v2.compare(v3), 0);

  // If there is mismatch before '\0', should still handle it.
  string v4 = "acc one";
  string v5 = "abb two";
  v4[3] = '\0';
  v5[3] = '\0';
  String str_v4{v4};
  String str_v5{v5};
  EXPECT_GT(str_v4.compare(str_v5), 0);
  EXPECT_EQ(str_v4.size(), 7);
  // strcmp is able to detect the mismatch
  EXPECT_GT(strcmp(v4.data(), v5.data()), 0);
  // string::compare can handle \0 since it knows size
  EXPECT_GT(v4.compare(v5), 0);
}

TEST(String, compare_same_memory_region_different_size) {
  using namespace std;
  string source = "a string";
  String str_source{source};
  char* memory = const_cast<char*>(str_source.data());
  EXPECT_EQ(str_source.compare(memory), 0);
  // This changes the string size
  memory[2] = '\0';
  // memory is logically shorter now
  EXPECT_GT(str_source.compare(memory), 0);
}

TEST(String, compare) {
  using namespace std;
  constexpr auto mismatch1_cstr = "a string but longer";
  string source = "a string";
  string mismatch1 = mismatch1_cstr;
  string mismatch2 = "a strin";
  string mismatch3 = "a b";
  string mismatch4 = "a t";
  String str_source{source};
  String str_mismatch1{mismatch1_cstr};
  String str_mismatch2{mismatch2};
  String str_mismatch3{mismatch3};
  String str_mismatch4{mismatch4};

  // compare with string
  EXPECT_EQ(str_source.compare(source), 0);
  EXPECT_TRUE(str_source == source);
  EXPECT_TRUE(source == str_source);
  EXPECT_TRUE(str_source <= source);
  EXPECT_TRUE(source <= str_source);
  EXPECT_TRUE(str_source >= source);
  EXPECT_TRUE(source >= str_source);
  EXPECT_LT(str_source.compare(mismatch1), 0);
  EXPECT_TRUE(str_source < mismatch1);
  EXPECT_TRUE(mismatch1 != str_source);
  EXPECT_GT(str_source.compare(mismatch2), 0);
  EXPECT_TRUE(str_source > mismatch2);
  EXPECT_TRUE(mismatch2 < str_source);
  EXPECT_GT(str_source.compare(mismatch3), 0);
  EXPECT_TRUE(str_source > mismatch3);
  EXPECT_LT(str_source.compare(mismatch4), 0);
  EXPECT_TRUE(str_source < mismatch4);
  EXPECT_TRUE(mismatch4 > str_source);

  // compare with char*
  EXPECT_EQ(str_source.compare(source.data()), 0);
  EXPECT_TRUE(str_source == source.data());
  EXPECT_TRUE(source.data() == str_source);
  EXPECT_TRUE(str_source <= source.data());
  EXPECT_TRUE(source <= str_source.data());
  EXPECT_TRUE(str_source >= source.data());
  EXPECT_TRUE(source >= str_source.data());
  EXPECT_LT(str_source.compare(mismatch1.data()), 0);
  EXPECT_TRUE(str_source < mismatch1.data());
  EXPECT_TRUE(str_source != mismatch1.data());
  EXPECT_TRUE(mismatch1.data() != str_source);
  EXPECT_GT(str_source.compare(mismatch2.data()), 0);
  EXPECT_TRUE(str_source > mismatch2.data());
  EXPECT_TRUE(mismatch2.data() < str_source);
  EXPECT_GT(str_source.compare(mismatch3.data()), 0);
  EXPECT_TRUE(str_source > mismatch3.data());
  EXPECT_LT(str_source.compare(mismatch4.data()), 0);
  EXPECT_TRUE(str_source < mismatch4.data());
  EXPECT_TRUE(mismatch4.data() > str_source);

  // compare with String
  EXPECT_LT(str_source.compare(str_mismatch1), 0);
  EXPECT_TRUE(str_source < str_mismatch1);
  EXPECT_GT(str_source.compare(str_mismatch2), 0);
  EXPECT_TRUE(str_source > str_mismatch2);
  EXPECT_GT(str_source.compare(str_mismatch3), 0);
  EXPECT_TRUE(str_source > str_mismatch3);
  EXPECT_LT(str_source.compare(str_mismatch4), 0);
  EXPECT_TRUE(str_source < str_mismatch4);
}

TEST(String, c_str) {
  using namespace std;
  string source = "this is a string";
  string mismatch = "mismatch";
  String s{source};

  EXPECT_EQ(std::strcmp(s.c_str(), source.data()), 0);
  EXPECT_NE(std::strcmp(s.c_str(), mismatch.data()), 0);
}

TEST(String, hash) {
  using namespace std;
  string source = "this is a string";
  String s{source};
  std::hash<String>()(s);

  std::unordered_map<String, std::string> map;
  String k1{string{"k1"}};
  string v1{"v1"};
  String k2{string{"k2"}};
  string v2{"v2"};
  map[k1] = v1;
  map[k2] = v2;

  EXPECT_EQ(map[k1], v1);
  EXPECT_EQ(map[k2], v2);
}

TEST(String, Cast) {
  using namespace std;
  string source = "this is a string";
  String s{source};
  ObjectRef r = s;
  String s2 = Downcast<String>(r);
}

TEST(String, Concat) {
  String s1("hello");
  String s2("world");
  std::string s3("world");
  String res1 = s1 + s2;
  String res2 = s1 + s3;
  String res3 = s3 + s1;
  String res4 = s1 + "world";
  String res5 = "world" + s1;

  EXPECT_EQ(res1.compare("helloworld"), 0);
  EXPECT_EQ(res2.compare("helloworld"), 0);
  EXPECT_EQ(res3.compare("worldhello"), 0);
  EXPECT_EQ(res4.compare("helloworld"), 0);
  EXPECT_EQ(res5.compare("worldhello"), 0);
}

TEST(String, Any) {
  // test anyview promotion to any
  AnyView view = "hello";

  Any b = view;
  EXPECT_EQ(b.type_index(), TypeIndex::kTVMFFIStr);
  EXPECT_EQ(b.as<String>().value(), "hello");
  EXPECT_TRUE(b.as<String>().has_value());
  EXPECT_EQ(b.try_cast<std::string>().value(), "hello");

  std::string s_world = "world";
  view = s_world;
  EXPECT_EQ(view.try_cast<std::string>().value(), "world");

  String s{"hello"};
  Any a = s;
  EXPECT_EQ(a.type_index(), TypeIndex::kTVMFFIStr);
  EXPECT_EQ(a.as<String>().value(), "hello");
  EXPECT_EQ(a.try_cast<std::string>().value(), "hello");

  Any c = "helloworld";
  EXPECT_EQ(c.type_index(), TypeIndex::kTVMFFIStr);
  EXPECT_EQ(c.as<String>().value(), "helloworld");
  EXPECT_EQ(c.try_cast<std::string>().value(), "helloworld");
}

TEST(String, Bytes) {
  // explicitly test zero element
  std::string s = {'\0', 'a', 'b', 'c'};
  Bytes b = s;
  EXPECT_EQ(b.size(), 4);
  EXPECT_EQ(b.operator std::string(), s);

  TVMFFIByteArray arr{s.data(), static_cast<size_t>(s.size())};
  Bytes b2 = arr;
  EXPECT_EQ(b2.size(), 4);
  EXPECT_EQ(b2.operator std::string(), s);
}

TEST(String, BytesAny) {
  std::string s = {'\0', 'a', 'b', 'c'};
  TVMFFIByteArray arr{s.data(), static_cast<size_t>(s.size())};

  AnyView view = &arr;
  EXPECT_EQ(view.type_index(), TypeIndex::kTVMFFIByteArrayPtr);
  EXPECT_EQ(view.try_cast<Bytes>().value().operator std::string(), s);

  Any b = view;
  EXPECT_EQ(b.type_index(), TypeIndex::kTVMFFIBytes);

  EXPECT_EQ(b.try_cast<Bytes>().value().operator std::string(), s);
  EXPECT_EQ(b.cast<std::string>(), s);
}

TEST(String, StdString) {
  std::string s1 = "test_string";
  AnyView view1 = s1;
  EXPECT_EQ(view1.type_index(), TypeIndex::kTVMFFIRawStr);
  EXPECT_EQ(view1.try_cast<std::string>().value(), s1);

  TVMFFIByteArray arr1{s1.data(), static_cast<size_t>(s1.size())};
  AnyView view2 = &arr1;
  EXPECT_EQ(view2.type_index(), TypeIndex::kTVMFFIByteArrayPtr);
  EXPECT_EQ(view2.try_cast<std::string>().value(), s1);

  Bytes bytes1 = s1;
  AnyView view3 = bytes1;
  EXPECT_EQ(view3.type_index(), TypeIndex::kTVMFFIBytes);
  EXPECT_EQ(view3.try_cast<std::string>().value(), s1);

  String string1 = s1;
  AnyView view4 = string1;
  EXPECT_EQ(view4.type_index(), TypeIndex::kTVMFFIStr);
  EXPECT_EQ(view4.try_cast<std::string>().value(), s1);

  // Test with Any
  Any any1 = s1;
  EXPECT_EQ(any1.type_index(), TypeIndex::kTVMFFIStr);
  EXPECT_EQ(any1.try_cast<std::string>().value(), s1);

  Any any2 = &arr1;
  EXPECT_EQ(any2.type_index(), TypeIndex::kTVMFFIBytes);
  EXPECT_EQ(any2.try_cast<std::string>().value(), s1);

  Any any3 = bytes1;
  EXPECT_EQ(any3.type_index(), TypeIndex::kTVMFFIBytes);
  EXPECT_EQ(any3.try_cast<std::string>().value(), s1);

  Any any4 = string1;
  EXPECT_EQ(any4.type_index(), TypeIndex::kTVMFFIStr);
  EXPECT_EQ(any4.try_cast<std::string>().value(), s1);
}

TEST(String, CAPIAccessor) {
  using namespace std;
  String s{"hello"};
  TVMFFIObjectHandle obj = details::ObjectUnsafe::RawObjectPtrFromObjectRef(s);
  TVMFFIByteArray* arr = TVMFFIBytesGetByteArrayPtr(obj);
  EXPECT_EQ(arr->size, 5);
  EXPECT_EQ(std::string(arr->data, arr->size), "hello");
}
}  // namespace
