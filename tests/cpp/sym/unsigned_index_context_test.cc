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
#include <tvm/sym/analyzer.h>

#include <future>
#include <stdexcept>

namespace tvm {
namespace sym {

TEST(AllowUIntAsIndex, ScopedEntry) {
  AllowUIntAsIndex context;
  EXPECT_FALSE(context.Enabled());
  {
    With<AllowUIntAsIndex> outer;
    EXPECT_TRUE(AllowUIntAsIndex::Enabled());
    {
      With<AllowUIntAsIndex> inner;
      EXPECT_TRUE(AllowUIntAsIndex::Enabled());
    }
    EXPECT_TRUE(AllowUIntAsIndex::Enabled());
  }
  EXPECT_FALSE(AllowUIntAsIndex::Enabled());
}

TEST(AllowUIntAsIndex, ExceptionCleanup) {
  EXPECT_THROW(
      {
        With<AllowUIntAsIndex> scope;
        throw std::runtime_error("leave scope");
      },
      std::runtime_error);
  EXPECT_FALSE(AllowUIntAsIndex::Enabled());
}

TEST(AllowUIntAsIndex, ThreadIsolation) {
  With<AllowUIntAsIndex> scope;
  auto other = std::async(std::launch::async, []() {
    bool isolated = !AllowUIntAsIndex::Enabled();
    {
      With<AllowUIntAsIndex> local;
      isolated = isolated && AllowUIntAsIndex::Enabled();
    }
    return isolated && !AllowUIntAsIndex::Enabled();
  });
  EXPECT_TRUE(other.get());
  EXPECT_TRUE(AllowUIntAsIndex::Enabled());
}

}  // namespace sym
}  // namespace tvm
