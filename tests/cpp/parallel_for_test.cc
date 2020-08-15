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

#include "../../src/support/parallel_for.h"

#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include <vector>

TEST(ParallelFor, Basic) {
  using tvm::support::parallel_for;

  int a[100], b[100];

  // Default
  for (int i = 0; i < 100; i++) {
    a[i] = i;
  }

  parallel_for(0, 100, [&b](int i) { b[i] = i; });

  for (int i = 0; i < 100; i++) {
    CHECK_EQ(a[i], b[i]);
  }

  // Check for step != 1
  for (int i = 0; i < 100; i += 2) {
    a[i] *= 2;
  }

  parallel_for(
      0, 100, [&b](int i) { b[i] *= 2; }, 2);

  for (int i = 0; i < 100; i++) {
    CHECK_EQ(a[i], b[i]);
  }
}

TEST(ParallelFor, Nested) {
  using tvm::support::parallel_for;

  int a[100][100], b[100][100], c[100][100];

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) {
      a[i][j] = i * j;
    }
  }

  parallel_for(0, 100, [&b](int i) {
    for (int j = 0; j < 100; j++) {
      b[i][j] = i * j;
    }
  });

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) {
      CHECK_EQ(a[i][j], b[i][j]);
    }
  }

  for (int i = 0; i < 100; i++) {
    parallel_for(0, 100, [&c, &i](int j) { c[i][j] = i * j; });
  }

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) {
      CHECK_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(ParallelFor, Exception) {
  using tvm::support::parallel_for;

  bool exception = false;
  try {
    parallel_for(0, 100, [](int i) { LOG(FATAL) << "error"; });
  } catch (const std::exception& e) {
    exception = true;
  }
  CHECK(exception);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
