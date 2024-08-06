#include <gtest/gtest.h>
#include <tvm/ffi/c_ffi_abi.h>

namespace {

TEST(ABIHeaderAlignment, Default) {
  TVMFFIObject value;
  value.type_index = 10;
  EXPECT_EQ(reinterpret_cast<TVMFFIAny*>(&value)->type_index, 10);
}

}  // namespace
