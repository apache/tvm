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
#include <tvm/runtime/logging.h>

namespace tvm {
namespace runtime {
namespace detail {
namespace {

TEST(ParseTvmLogDebugSpec, Disabled) {
  EXPECT_TRUE(ParseTvmLogDebugSpec(nullptr).empty());
  EXPECT_TRUE(ParseTvmLogDebugSpec("").empty());
  EXPECT_TRUE(ParseTvmLogDebugSpec("0").empty());
}

TEST(ParseTvmLogDebugSpec, DlogOnly) {
  auto map = ParseTvmLogDebugSpec("1");
  EXPECT_EQ(map.size(), 1);
  EXPECT_EQ(map["*"], -1);
}

TEST(ParseTvmLogDebugSpec, VLogEnabled) {
  auto map = ParseTvmLogDebugSpec("foo/bar.cc=3;baz.cc=-1;*=2;another/file.cc=4");
  EXPECT_EQ(map.size(), 4);

  EXPECT_EQ(map["*"], 2);
  EXPECT_EQ(map["foo/bar.cc"], 3);
  EXPECT_EQ(map["baz.cc"], -1);
  EXPECT_EQ(map["another/file.cc"], 4);
}

TEST(ParseTvmLogDebugSpec, IllFormed) {
  EXPECT_THROW(ParseTvmLogDebugSpec("foo/bar.cc=bogus;"), InternalError);
}

TEST(VerboseEnabledInMap, Lookup) {
  auto map = ParseTvmLogDebugSpec("foo/bar.cc=3;baz.cc=-1;*=2;another/file.cc=4;");

  EXPECT_TRUE(VerboseEnabledInMap("my/filesystem/src/foo/bar.cc", 3, map));
  EXPECT_FALSE(VerboseEnabledInMap("my/filesystem/src/foo/bar.cc", 4, map));
  EXPECT_TRUE(VerboseEnabledInMap("my/filesystem/src/foo/other.cc", 2, map));
  EXPECT_FALSE(VerboseEnabledInMap("my/filesystem/src/foo/other.cc", 3, map));
  EXPECT_FALSE(VerboseEnabledInMap("my/filesystem/src/baz.cc", 0, map));
}

}  // namespace
}  // namespace detail
}  // namespace runtime
}  // namespace tvm
