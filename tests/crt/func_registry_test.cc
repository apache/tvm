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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/crt/func_registry.h>
#include <tvm/runtime/crt/internal/common/func_registry.h>

typedef struct {
  const char* a;
  const char* b;
  int ret_val;
} strcmp_test_t;

strcmp_test_t strcmp_tests[] = {
    {"Foo", "Foo", 0},        {"Foo", "Bar", 'F' - 'B'},    {"Foo", "", 'F'},
    {"Fabulous", "Fab", 'u'}, {"Fab", "Fabulous", 0 - 'u'},
};

std::ostream& operator<<(std::ostream& os, const strcmp_test_t& test) {
  os << "strcmp_cursor(\"" << test.a << "\", \"" << test.b << "\") -> " << test.ret_val;
  return os;
}

class StrCmpTestFixture : public ::testing::TestWithParam<strcmp_test_t> {};

TEST_P(StrCmpTestFixture, Match) {
  strcmp_test_t param = GetParam();
  const char* cursor = param.a;
  EXPECT_EQ(param.ret_val, strcmp_cursor(&cursor, param.b));

  EXPECT_EQ('\0', *cursor);

  size_t a_length = strlen(param.a);
  EXPECT_EQ(param.a + a_length, cursor);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
INSTANTIATE_TEST_CASE_P(StrCmpTests, StrCmpTestFixture, ::testing::ValuesIn(strcmp_tests));
#pragma GCC diagnostic pop

TEST(StrCmpScan, Test) {
  const char* a = "Foo\0Bar\0Whoops\0";
  const char* cursor = a;

  EXPECT_EQ('o', strcmp_cursor(&cursor, "Fo"));
  EXPECT_EQ(0, *cursor);
  EXPECT_EQ(cursor, a + 3);
  cursor++;

  EXPECT_EQ(0 - 'r', strcmp_cursor(&cursor, "Barr"));
  EXPECT_EQ(0, *cursor);
  EXPECT_EQ(cursor, a + 7);
  cursor++;

  EXPECT_EQ('h' - 'B', strcmp_cursor(&cursor, "WB"));
  EXPECT_EQ(0, *cursor);
  EXPECT_EQ(cursor, a + 14);
  cursor++;

  EXPECT_EQ(0, *cursor);
  const char* before_cursor = cursor;
  EXPECT_EQ(0, strcmp_cursor(&cursor, ""));
  EXPECT_EQ(before_cursor, cursor);
}

TEST(FuncRegistry, Empty) {
  TVMFuncRegistry registry{"\000\000", NULL};

  EXPECT_EQ(kTvmErrorFunctionNameNotFound, TVMFuncRegistry_Lookup(&registry, "foo", NULL));
  EXPECT_EQ(kTvmErrorFunctionIndexInvalid,
            TVMFuncRegistry_GetByIndex(&registry, (tvm_function_index_t)0, NULL));
}

extern "C" {
static int Foo(TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,
               int* out_ret_tcode, void* resource_handle) {
  return 0;
}
static int Bar(TVMValue* args, int* type_codes, int num_args, TVMValue* out_ret_value,
               int* out_ret_tcode, void* resource_handle) {
  return 0;
}
}

// Matches the style of registry defined in generated C modules.
const char* kBasicFuncNames = "\002\000Foo\0Bar\0";  // NOTE: final \0
const TVMBackendPackedCFunc funcs[2] = {&Foo, &Bar};
const TVMFuncRegistry kConstRegistry = {kBasicFuncNames, (const TVMBackendPackedCFunc*)funcs};

TEST(FuncRegistry, ConstGlobalRegistry) {
  tvm_function_index_t func_index = -1;
  TVMBackendPackedCFunc func = nullptr;

  // Foo
  EXPECT_EQ(kBasicFuncNames[0], 2);
  EXPECT_EQ(kBasicFuncNames[1], 0);
  EXPECT_EQ(kBasicFuncNames[2], 'F');
  EXPECT_EQ(kTvmErrorNoError, TVMFuncRegistry_Lookup(&kConstRegistry, "Foo", &func_index));
  EXPECT_EQ(0, func_index);

  EXPECT_EQ(kTvmErrorNoError, TVMFuncRegistry_GetByIndex(&kConstRegistry, func_index, &func));
  EXPECT_EQ(func, &Foo);

  // Bar
  EXPECT_EQ(kTvmErrorNoError, TVMFuncRegistry_Lookup(&kConstRegistry, "Bar", &func_index));
  EXPECT_EQ(1, func_index);

  EXPECT_EQ(kTvmErrorNoError, TVMFuncRegistry_GetByIndex(&kConstRegistry, func_index, &func));
  EXPECT_EQ(func, &Bar);

  // Expected not found.
  tvm_function_index_t prev_func_index = func_index;
  EXPECT_EQ(kTvmErrorFunctionNameNotFound,
            TVMFuncRegistry_Lookup(&kConstRegistry, "Baz", &func_index));
  EXPECT_EQ(prev_func_index, func_index);

  // Expected index out of range.
  func = nullptr;
  EXPECT_EQ(kTvmErrorFunctionIndexInvalid, TVMFuncRegistry_GetByIndex(&kConstRegistry, 2, &func));
  EXPECT_EQ(func, nullptr);
}

/*! \brief Return a test function handle, with number repeating for all bytes in a void*. */
static TVMBackendPackedCFunc TestFunctionHandle(uint8_t number) {
  uintptr_t handle = 0;
  for (size_t i = 0; i < sizeof(TVMBackendPackedCFunc); i++) {
    handle |= ((uintptr_t)handle) << (8 * i);
  }

  return (TVMBackendPackedCFunc)handle;
}

static void snprintf_truncate(char* target, size_t bytes, const char* str) {
#ifdef __GNUC__
#if __GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ >= 1)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
#endif
  EXPECT_GT(snprintf(target, bytes, "%s", str), 0);
#ifdef __GNUC__
#if __GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ >= 1)
#pragma GCC diagnostic pop
#endif
#endif
}

TEST(MutableFuncRegistry, Create) {
  uint8_t mem_buffer[kTvmAverageFuncEntrySizeBytes * 3];
  // A substring used to create function names for testing.
  const char* function_name_chars = "abcdefghijklmnopqrstuvwxyzyxw";

  // function_name_chars is used to produce 2 function names. The second one is expected to
  // overfill `names`; assert there are at least enough data in function_name_chars to do this.
  EXPECT_LE(kTvmAverageFuncEntrySizeBytes + kTvmAverageFunctionNameStrlenBytes,
            strlen(function_name_chars));

  for (unsigned int buf_size = 0; buf_size < kTvmAverageFuncEntrySizeBytes; buf_size++) {
    EXPECT_EQ(kTvmErrorBufferTooSmall, TVMMutableFuncRegistry_Create(NULL, mem_buffer, buf_size));
  }

  for (unsigned int rem = 0; rem < kTvmAverageFuncEntrySizeBytes; rem++) {
    // test_function name will be used to test overfilling.
    auto test_function_name =
        std::make_unique<char[]>(kTvmAverageFunctionNameStrlenBytes + 2 + rem);
    TVMMutableFuncRegistry reg;
    memset(mem_buffer, 0, sizeof(mem_buffer));
    EXPECT_EQ(kTvmErrorNoError, TVMMutableFuncRegistry_Create(
                                    &reg, mem_buffer, kTvmAverageFuncEntrySizeBytes * 2 + rem));

    snprintf_truncate(test_function_name.get(), kTvmAverageFunctionNameStrlenBytes + 1,
                      function_name_chars);

    // Add function #1, and verify it can be retrieved.
    EXPECT_EQ(kTvmErrorNoError, TVMMutableFuncRegistry_Set(&reg, test_function_name.get(),
                                                           TestFunctionHandle(0x01), 0));

    tvm_function_index_t func_index = 100;
    EXPECT_EQ(kTvmErrorNoError,
              TVMFuncRegistry_Lookup(&reg.registry, test_function_name.get(), &func_index));
    EXPECT_EQ(func_index, 0);

    TVMBackendPackedCFunc func = NULL;
    EXPECT_EQ(kTvmErrorNoError, TVMFuncRegistry_GetByIndex(&reg.registry, func_index, &func));
    EXPECT_EQ(func, TestFunctionHandle(0x01));

    // Ensure that overfilling `names` by 1 char is not allowed.
    snprintf_truncate(test_function_name.get(), kTvmAverageFunctionNameStrlenBytes + rem + 2,
                      function_name_chars + 1);

    EXPECT_EQ(
        kTvmErrorFunctionRegistryFull,
        TVMMutableFuncRegistry_Set(&reg, test_function_name.get(), TestFunctionHandle(0x02), 0));
    EXPECT_EQ(kTvmErrorFunctionNameNotFound,
              TVMFuncRegistry_Lookup(&reg.registry, test_function_name.get(), &func_index));

    // Add function #2, with intentionally short (by 2 char) name. Verify it can be retrieved.
    snprintf_truncate(test_function_name.get(), kTvmAverageFunctionNameStrlenBytes - 2 + 1,
                      function_name_chars + 1);
    EXPECT_EQ(kTvmErrorNoError, TVMMutableFuncRegistry_Set(&reg, test_function_name.get(),
                                                           TestFunctionHandle(0x02), 0));

    EXPECT_EQ(kTvmErrorNoError,
              TVMFuncRegistry_Lookup(&reg.registry, test_function_name.get(), &func_index));
    EXPECT_EQ(func_index, 1);

    func = NULL;
    EXPECT_EQ(kTvmErrorNoError, TVMFuncRegistry_GetByIndex(&reg.registry, func_index, &func));
    EXPECT_EQ(func, TestFunctionHandle(0x01));

    // Try adding another function, which should fail due to lack of function pointers.
    test_function_name[0] = 'a';
    test_function_name[1] = 0;
    EXPECT_EQ(
        kTvmErrorFunctionRegistryFull,
        TVMMutableFuncRegistry_Set(&reg, test_function_name.get(), TestFunctionHandle(0x03), 0));
  }
}
