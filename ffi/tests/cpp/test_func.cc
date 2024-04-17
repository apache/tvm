#include <gtest/gtest.h>
#include <tvm/ffi/ffi.hpp>

namespace {
using namespace tvm::ffi;

const char *c_str_raw = "Hello";

double func_unpacked_0(int64_t a, double b, const char *c, const double &d) {
  EXPECT_STREQ(c, c_str_raw);
  return a + b + d;
}

void func_unpacked_1(DLDataType dtype, DLDevice device, std::string str) {
  (void)dtype;
  (void)device;
  (void)str;
}

void func_packed_0(int num_args, const AnyView *, Any *ret) { *ret = num_args; }

template <enum TVMFFITypeIndex type_index>
void func_unpacked_anyview_arg(AnyView a) {
  EXPECT_EQ(a.type_index, static_cast<int32_t>(type_index));
}
template <enum TVMFFITypeIndex type_index>
void func_unpacked_any_arg(Any a) {
  EXPECT_EQ(a.type_index, static_cast<int32_t>(type_index));
}
AnyView func_unpacked_anyview_ret() { return AnyView(1); }
Any func_unpacked_any_ret() { return Any(1); }

std::string func_unpacked_str_obj(Str *str, const char *str_2) {
  EXPECT_EQ(reinterpret_cast<TVMFFIStr *>(str)->ref_cnt, 1);
  EXPECT_STREQ(str->c_str(), str_2);
  return str->c_str();
}

TEST(Func_Signature, 0) {
  EXPECT_EQ(details::FuncFunctor<decltype(func_unpacked_0)>::Sig(),
            "(0: int, 1: float, 2: const char *, 3: float) -> float");
}

TEST(Func_Signature, 1) {
  EXPECT_EQ(details::FuncFunctor<decltype(func_unpacked_1)>::Sig(),
            "(0: dtype, 1: Device, 2: str) -> void");
}

TEST(Func_Signature, AnyView_Arg) {
  EXPECT_EQ(details::FuncFunctor<decltype(func_unpacked_anyview_arg<
                                          TVMFFITypeIndex::kTVMFFIInt>)>::Sig(),
            "(0: AnyView) -> void");
}

TEST(Func_Signature, AnyView_Ret) {
  EXPECT_EQ(details::FuncFunctor<decltype(func_unpacked_anyview_ret)>::Sig(),
            "() -> AnyView");
}

TEST(Func_Signature, Any_Arg) {
  EXPECT_EQ(
      details::FuncFunctor<
          decltype(func_unpacked_any_arg<TVMFFITypeIndex::kTVMFFIInt>)>::Sig(),
      "(0: Any) -> void");
}

TEST(Func_Signature, Any_Ret) {
  EXPECT_EQ(details::FuncFunctor<decltype(func_unpacked_any_ret)>::Sig(),
            "() -> Any");
}

TEST(Func_Unpacked_Invoke, Func0_RawStr) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_0);
  double ret = func(1, 2, c_str_raw, 4);
  EXPECT_DOUBLE_EQ(ret, 7);
  double ret2 = func(1, 2, std::string("Hello"), 4);
  EXPECT_DOUBLE_EQ(ret2, 7);
}

TEST(Func_Unpacked_Invoke, Func0_StdString_Move) {
  std::string str = c_str_raw;
  Ref<Func> func = Ref<Func>::New(func_unpacked_0);
  double ret = func(1, 2, std::move(str), 4);
  EXPECT_DOUBLE_EQ(ret, 7);
}

TEST(Func_Unpacked_Invoke, Func0_StdString_Copy) {
  std::string str = c_str_raw;
  Ref<Func> func = Ref<Func>::New(func_unpacked_0);
  double ret = func(1, 2, str, 4);
  EXPECT_DOUBLE_EQ(ret, 7);
  EXPECT_EQ(str, c_str_raw);
}

TEST(Func_Unpacked_Invoke, Func1) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_1);
  func(DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0}, "Hello");
}

TEST(Func_Unpacked_Invoke, AnyView_Arg) {
  Ref<Func> func =
      Ref<Func>::New(func_unpacked_anyview_arg<TVMFFITypeIndex::kTVMFFIInt>);
  func(1);
}

TEST(Func_Unpacked_Invoke, AnyView_Ret) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_anyview_ret);
  int ret = func();
  EXPECT_EQ(ret, 1);
}

TEST(Func_Unpacked_Invoke, Any_Arg) {
  Ref<Func> func =
      Ref<Func>::New(func_unpacked_any_arg<TVMFFITypeIndex::kTVMFFIInt>);
  func(1);
}

TEST(Func_Unpacked_Invoke, Any_Ret) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_any_ret);
  int ret = func();
  EXPECT_EQ(ret, 1);
}

TEST(Func_Unpacked_Invoke, StrObj) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_str_obj);
  std::string ret = func("Hello", "Hello");
  EXPECT_EQ(ret, "Hello");
}

TEST(Func_Packed_Invoke, 0) {
  Ref<Func> func = Ref<Func>::New(func_packed_0);
  int ret = func();
  EXPECT_EQ(ret, 0);
}

TEST(Func_Packed_Invoke, 1) {
  Ref<Func> func = Ref<Func>::New(func_packed_0);
  int ret = func(1.0);
  EXPECT_EQ(ret, 1);
}

TEST(Func_Packed_Invoke, 2) {
  Ref<Func> func = Ref<Func>::New(func_packed_0);
  int ret = func(1.0, "test");
  EXPECT_EQ(ret, 2);
}

TEST(Func_Packed_Invoke, 4) {
  Ref<Func> func = Ref<Func>::New(func_packed_0);
  int ret = func(1.0, "test", DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0});
  EXPECT_EQ(ret, 4);
}

TEST(Func_Unpacked_Invoke_TypeError, TypeMismatch_0) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_0);
  try {
    func(1.0, 2, c_str_raw, 4);
    FAIL() << "No execption thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(),
                 "Mismatched type on argument #0 when calling: "
                 "`(0: int, 1: float, 2: const char *, 3: float) -> float`. "
                 "Expected `int` but got `float`");
  }
}

TEST(Func_Unpacked_Invoke_TypeError, TypeMismatch_1) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_1);
  try {
    func(DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0}, 1);
    FAIL() << "No execption thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(), "Mismatched type on argument #2 when calling: "
                            "`(0: dtype, 1: Device, 2: str) -> void`. "
                            "Expected `str` but got `int`");
  }
}

TEST(Func_Unpacked_Invoke_TypeError, ArgCountMismatch_0) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_0);
  try {
    func(1, 2, c_str_raw);
    FAIL() << "No execption thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(),
                 "Mismatched number of arguments when calling: "
                 "`(0: int, 1: float, 2: const char *, 3: float) -> float`. "
                 "Expected 4 but got 3 arguments");
  }
}

TEST(Func_Unpacked_Invoke_TypeError, ArgCountMismatch_1) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_1);
  try {
    func(DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0});
    FAIL() << "No execption thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(), "Mismatched number of arguments when calling: "
                            "`(0: dtype, 1: Device, 2: str) -> void`. "
                            "Expected 3 but got 2 arguments");
  }
}

TEST(Func_Unpacked_Invoke_TypeError, ReturnTypeMismatch_0) {
  Ref<Func> func = Ref<Func>::New(func_unpacked_1);
  try {
    int ret = func(DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0}, "Hello");
    (void)ret;
    FAIL() << "No execption thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(), "Cannot convert from type `None` to `int`");
  }
}

} // namespace
