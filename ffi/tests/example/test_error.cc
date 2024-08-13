#include <gtest/gtest.h>
#include <tvm/ffi/error.h>

namespace {

using namespace tvm::ffi;

void ThrowRuntimeError() {
  TVM_FFI_THROW(RuntimeError)
    << "test0";
}

TEST(Error, Traceback) {
  EXPECT_THROW({
    try {
      ThrowRuntimeError();
    } catch (const Error& error) {
      EXPECT_EQ(error->message, "test0");
      EXPECT_EQ(error->kind, "RuntimeError");
      std::string what = error.what();
      EXPECT_NE(what.find("line"), std::string::npos);
      EXPECT_NE(what.find("ThrowRuntimeError"), std::string::npos);
      EXPECT_NE(what.find("RuntimeError: test0"), std::string::npos);
      throw;
    }
  }, ::tvm::ffi::Error);
}

TEST(CheckError, Traceback) {
  EXPECT_THROW(
      {
        try {
          TVM_FFI_ICHECK_GT(2, 3);
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "InternalError");
          std::cout << error.what();
          std::string what = error.what();
          EXPECT_NE(what.find("line"), std::string::npos);
          EXPECT_NE(what.find("2 > 3"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}
}  // namespace
