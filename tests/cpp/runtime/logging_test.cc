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
#include <tvm/runtime/logging.h>

namespace tvm {
namespace runtime {
namespace detail {
namespace {

TEST(ParseTvmLogDebugSpec, Disabled) {
  auto map = ParseTvmLogDebugSpec(nullptr);
  EXPECT_THAT(map.size(), testing::Eq(0));

  map = ParseTvmLogDebugSpec("");
  EXPECT_THAT(map.size(), testing::Eq(0));

  map = ParseTvmLogDebugSpec("0");
  EXPECT_THAT(map.size(), testing::Eq(0));

  map = ParseTvmLogDebugSpec("1");
  EXPECT_THAT(map.size(), testing::Eq(0));
}

TEST(ParseTvmLogDebugSpec, SomeEnabled) {
  auto map = ParseTvmLogDebugSpec("1;foo/bar.cc=3;baz.cc=-1;*=2;another/file.cc=4;");
  EXPECT_THAT(map.size(), testing::Eq(4));

  EXPECT_THAT(map, testing::Contains(testing::Pair("*", 2)));
  EXPECT_THAT(map, testing::Contains(testing::Pair("foo/bar.cc", 3)));
  EXPECT_THAT(map, testing::Contains(testing::Pair("baz.cc", -1)));
  EXPECT_THAT(map, testing::Contains(testing::Pair("another/file.cc", 4)));
}

TEST(ParseTvmLogDebugSpec, IllFormed) {
  EXPECT_THROW(ParseTvmLogDebugSpec("1;foo/bar.cc=bogus;"), InternalError);
}

TEST(VerboseEnabledInMap, Lookup) {
  auto map = ParseTvmLogDebugSpec("1;foo/bar.cc=3;baz.cc=-1;*=2;another/file.cc=4;");

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
