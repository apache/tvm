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
#include <tvm/support/parallel_for.h>

#include <vector>

TEST(ParallelFor, Basic) {
  using tvm::support::parallel_for;

  int a[1000], b[1000];

  // Check for a small size of parallel
  for (int i = 0; i < 10; i++) {
    a[i] = i;
  }
  parallel_for(0, 10, [&b](int i) { b[i] = i; });
  for (int i = 0; i < 10; i++) {
    CHECK_EQ(a[i], b[i]);
  }

  // Check for a large size of parallel
  for (int i = 0; i < 1000; i++) {
    a[i] = i;
  }
  parallel_for(0, 1000, [&b](int i) { b[i] = i; });
  for (int i = 0; i < 1000; i++) {
    CHECK_EQ(a[i], b[i]);
  }

  // Check for step != 1
  for (int i = 0; i < 1000; i += 2) {
    a[i] *= 2;
  }
  parallel_for(
      0, 1000, [&b](int i) { b[i] *= 2; }, 2);
  for (int i = 0; i < 1000; i++) {
    CHECK_EQ(a[i], b[i]);
  }
}

TEST(ParallelFor, NestedWithNormalForLoop) {
  using tvm::support::parallel_for;

  int a[500][500], b[500][500], c[500][500];

  for (int i = 0; i < 500; i++) {
    for (int j = 0; j < 500; j++) {
      a[i][j] = i * j;
    }
  }

  parallel_for(0, 500, [&b](int i) {
    for (int j = 0; j < 500; j++) {
      b[i][j] = i * j;
    }
  });
  for (int i = 0; i < 500; i++) {
    for (int j = 0; j < 500; j++) {
      CHECK_EQ(a[i][j], b[i][j]);
    }
  }

  for (int i = 0; i < 500; i++) {
    parallel_for(0, 500, [&c, &i](int j) { c[i][j] = i * j; });
  }
  for (int i = 0; i < 500; i++) {
    for (int j = 0; j < 500; j++) {
      CHECK_EQ(a[i][j], c[i][j]);
    }
  }
}

TEST(Parallelfor, NestedWithParallelFor) {
  // Currently do not support using nested parallel_for
  using tvm::support::parallel_for;

  bool exception = false;
  try {
    parallel_for(0, 100, [](int i) {
      parallel_for(0, 100, [](int j) {
        // Blank loop
      });
    });
  } catch (const std::exception& e) {
    exception = true;
  }
  CHECK(exception);
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
