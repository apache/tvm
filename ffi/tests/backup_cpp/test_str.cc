#include <gtest/gtest.h>

#include <tvm/ffi/ffi.hpp>

namespace {
using namespace tvm::ffi;

const char c_str_long[] =
    "Hello, World! This is an extremely long string to "
    "avoid any on-stack optimization.";

TEST(Str, CopyFromStdString) {
  std::string std_str = "Hello, World!";
  Ref<Str> str = Ref<Str>::New(std_str);
  EXPECT_EQ(str->size(), std_str.size());
  EXPECT_STREQ(str->c_str(), std_str.c_str());
  EXPECT_STREQ(str->data(), std_str.data());
}

TEST(Str, MoveFromStdString_0) {
  std::string std_str = c_str_long;
  const void* data = std_str.data();
  Ref<Str> str = Ref<Str>::New(std::move(std_str));
  EXPECT_EQ(static_cast<const void*>(str->data()), data);
  EXPECT_STREQ(str->c_str(), c_str_long);
}

TEST(Str, MoveFromStdString_1) {
  Ref<Str> str = Ref<Str>::New(std::string(c_str_long));
  EXPECT_STREQ(str->c_str(), c_str_long);
  EXPECT_EQ(str->size(), sizeof(c_str_long) - 1);
}

TEST(Str, CopyFromCStr) {
  Ref<Str> str = Ref<Str>::New(c_str_long);
  EXPECT_STREQ(str->c_str(), c_str_long);
  EXPECT_EQ(str->size(), sizeof(c_str_long) - 1);
}

TEST(Str, CopyFromCharArray) {
  const char c_str[18] = "Hello, World!";
  Ref<Str> str = Ref<Str>::New(c_str);
  EXPECT_EQ(sizeof(c_str), 18);
  EXPECT_EQ(strlen(c_str), 13);
  EXPECT_STREQ(str->c_str(), c_str);
  EXPECT_EQ(str->size(), 17);
}

}  // namespace
