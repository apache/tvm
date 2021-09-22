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

TEST(TvmLogDebugSettings, Disabled) {
  TvmLogDebugSettings settings = TvmLogDebugSettings::ParseSpec(nullptr);
  EXPECT_FALSE(settings.dlog_enabled());

  settings = TvmLogDebugSettings::ParseSpec("");
  EXPECT_FALSE(settings.dlog_enabled());

  settings = TvmLogDebugSettings::ParseSpec("0");
  EXPECT_FALSE(settings.dlog_enabled());
}

TEST(TvmLogDebugSettings, DlogOnly) {
  TvmLogDebugSettings settings = TvmLogDebugSettings::ParseSpec("1");
  EXPECT_TRUE(settings.dlog_enabled());
  EXPECT_FALSE(settings.VerboseEnabled("my/filesytem/src/foo/bar.cc", 0));
}

TEST(TvmLogDebugSettings, VLogEnabledDefault) {
  TvmLogDebugSettings settings = TvmLogDebugSettings::ParseSpec("DEFAULT=3");
  EXPECT_TRUE(settings.dlog_enabled());
  EXPECT_TRUE(settings.VerboseEnabled("my/filesytem/src/foo/bar.cc", 3));
  EXPECT_FALSE(settings.VerboseEnabled("my/filesytem/src/foo/bar.cc", 4));
}

TEST(TvmLogDebugSettings, VLogEnabledComplex) {
  TvmLogDebugSettings settings =
      TvmLogDebugSettings::ParseSpec("foo/bar.cc=3;baz.cc=-1;DEFAULT=2;another/file.cc=4");
  EXPECT_TRUE(settings.dlog_enabled());
  EXPECT_TRUE(settings.VerboseEnabled("my/filesystem/src/foo/bar.cc", 3));
  EXPECT_FALSE(settings.VerboseEnabled("my/filesystem/src/foo/bar.cc", 4));
  EXPECT_TRUE(settings.VerboseEnabled("my/filesystem/src/foo/other.cc", 2));
  EXPECT_FALSE(settings.VerboseEnabled("my/filesystem/src/foo/other.cc", 3));
  EXPECT_FALSE(settings.VerboseEnabled("my/filesystem/src/baz.cc", 0));
}

TEST(TvmLogDebugSettings, IllFormed) {
  EXPECT_THROW(TvmLogDebugSettings::ParseSpec("foo/bar.cc=bogus;"), InternalError);
}

}  // namespace
}  // namespace detail
}  // namespace runtime
}  // namespace tvm
