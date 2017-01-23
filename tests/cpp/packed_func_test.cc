#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/runtime.h>

TEST(PackedFunc, Basic) {
  using namespace tvm::runtime;
  int x = 0;
  void* handle = &x;
  TVMArray a;

  PackedFunc([&](const TVMValue* args, const int* type_codes, int num_args) {
      CHECK(num_args == 3);
      CHECK(args[0].v_float64 == 1.0);
      CHECK(type_codes[0] == kFloat);
      CHECK(args[1].v_handle == &a);
      CHECK(type_codes[1] == kHandle);
      CHECK(args[2].v_handle == &x);
      CHECK(type_codes[2] == kHandle);
    })(1.0, &a, handle);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
