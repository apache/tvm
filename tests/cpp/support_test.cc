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

#include "../../src/support/hexdump.h"
#include "../../src/support/utils.h"

namespace tvm {
namespace test {

TEST(HexDumpTests, Empty) { EXPECT_EQ("", ::tvm::support::HexDump("")); }

TEST(HexDumpTests, Aligned) {
  EXPECT_EQ(
      "0000   01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef  .#Eg.....#Eg....\n"
      "0010   01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef  .#Eg.....#Eg....\n",
      ::tvm::support::HexDump("\x01\x23\x45\x67\x89\xab\xcd\xef\x01\x23\x45\x67\x89\xab\xcd\xef"
                              "\x01\x23\x45\x67\x89\xab\xcd\xef\x01\x23\x45\x67\x89\xab\xcd\xef"));
}

TEST(HexDumpTests, Unaligned) {
  EXPECT_EQ(
      "0000   01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef  .#Eg.....#Eg....\n"
      "0010   01 23 45 67 89 ab cd ef 01                       .#Eg.....\n",
      ::tvm::support::HexDump("\x01\x23\x45\x67\x89\xab\xcd\xef\x01\x23\x45\x67\x89\xab\xcd\xef"
                              "\x01\x23\x45\x67\x89\xab\xcd\xef\x01"));
}

TEST(HashTests, HashStability) {
  size_t a = 345292;
  int b = 795620;
  EXPECT_EQ(::tvm::support::HashCombine(a, b), 2677237020);
  uint64_t c = 12345;
  int d = 987654432;
  EXPECT_EQ(::tvm::support::HashCombine(c, d), 3642871070);
  size_t e = 1010101;
  size_t f = 3030303;
  EXPECT_EQ(::tvm::support::HashCombine(e, f), 2722928432);
}

TEST(StartsWithTests, Basic) {
  EXPECT_TRUE(::tvm::support::StartsWith("abc", "abc"));
  EXPECT_TRUE(::tvm::support::StartsWith("abcd", "abc"));
  EXPECT_FALSE(::tvm::support::StartsWith("abc", "abcd"));
}

}  // namespace test
}  // namespace tvm
