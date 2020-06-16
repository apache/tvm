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

#define TVM_CRT_LOG_VIRT_MEM_SIZE 16
#define TVM_CRT_PAGE_BYTES 4096

#include <gtest/gtest.h>
#include <tvm/runtime/crt/memory.h>

#include "../../src/runtime/crt/memory.c"

TEST(CRTMemory, Alloc) {
  for (int idx = 0; idx < 65536; idx++) {
    void* a = vmalloc(1);
    EXPECT_EQ(vleak_size, 1);
    vfree(a);
    EXPECT_EQ(vleak_size, 0);
  }
}

TEST(CRTMemory, Realloc) {
  for (int idx = 0; idx < 65536; idx++) {
    void* a = vrealloc(0, 1);
    EXPECT_EQ(vleak_size, 1);
    void* b = vrealloc(a, 1);
    EXPECT_EQ(a, b);
    EXPECT_EQ(vleak_size, 1);
    vfree(a);
    EXPECT_EQ(vleak_size, 0);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
