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
      TvmLogDebugSettings::ParseSpec("foo/bar.cc=3,baz.cc=-1,DEFAULT=2,another/file.cc=4");
  EXPECT_TRUE(settings.dlog_enabled());
  EXPECT_TRUE(settings.VerboseEnabled("my/filesystem/src/foo/bar.cc", 3));
  EXPECT_FALSE(settings.VerboseEnabled("my/filesystem/src/foo/bar.cc", 4));
  EXPECT_TRUE(settings.VerboseEnabled("my/filesystem/src/foo/other.cc", 2));
  EXPECT_FALSE(settings.VerboseEnabled("my/filesystem/src/foo/other.cc", 3));
  EXPECT_FALSE(settings.VerboseEnabled("my/filesystem/src/baz.cc", 0));
}

#define MATCH_THROW(stmt, err_type, matcher)            \
  try {                                                 \
    stmt;                                               \
  } catch (const err_type& e) {                         \
    EXPECT_THAT(e.what(), matcher);                     \
  } catch (...) {                                       \
    EXPECT_FALSE("stmt threw an unexpected exception"); \
  }

TEST(TvmLogDebugSettings, IllFormed) {
  MATCH_THROW(
      TvmLogDebugSettings::ParseSpec("foo/bar.cc=bogus;"), InternalError,
      ::testing::HasSubstr("TVM_LOG_DEBUG ill-formed at position 11: invalid level: \"bogus;\""));

  MATCH_THROW(TvmLogDebugSettings::ParseSpec("DEFAULT=2;bar/baz.cc=2"), InternalError,
              ::testing::HasSubstr(
                  "TVM_LOG_DEBUG ill-formed at position 8: invalid level: \"2;bar/baz.cc=2\""));

  MATCH_THROW(TvmLogDebugSettings::ParseSpec("DEFAULT=2,bar/baz.cc+2"), InternalError,
              ::testing::HasSubstr("TVM_LOG_DEBUG ill-formed at position 22: expecting "
                                   "\"=<level>\" after \"bar/baz.cc+2\""));
}

TEST(TvmLogDebugSettings, SpecPrefix) {
  TvmLogDebugSettings settings = TvmLogDebugSettings::ParseSpec(
      "../src/foo/bar.cc=3,src/baz.cc=3,foo/bar/src/another/file.cc=4");
  EXPECT_TRUE(settings.dlog_enabled());
  EXPECT_TRUE(settings.VerboseEnabled("my/filesystem/src/foo/bar.cc", 3));
  EXPECT_TRUE(settings.VerboseEnabled("foo/bar.cc", 3));
  EXPECT_TRUE(settings.VerboseEnabled("my/filesystem/src/baz.cc", 3));
  EXPECT_TRUE(settings.VerboseEnabled("baz.cc", 3));
  EXPECT_TRUE(settings.VerboseEnabled("my/filesystem/src/another/file.cc", 4));
  EXPECT_TRUE(settings.VerboseEnabled("another/file.cc", 4));
}

}  // namespace
}  // namespace detail
}  // namespace runtime
}  // namespace tvm
