#include "../src/support/parallel_for.h"

#include <dmlc/logging.h>
#include <gtest/gtest.h>

#include <vector>

TEST(ParallelFor, Basic) {
  using namespace tvm::support;

  int a[100], b[100];

  for (int i = 0; i < 100; i++) {
    a[i] = i;
  }

  parallel_for(0, 100, [&b](int i) { b[i] = i; });

  for (int i = 0; i < 100; i++) {
    CHECK_EQ(a[i], b[i]);
  }
}

TEST(ParallelFor, Nested) {
  using namespace tvm::support;

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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
