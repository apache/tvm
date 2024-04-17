#include <gtest/gtest.h>
#include <tvm/ffi/ffi.hpp>

namespace {
using namespace tvm::ffi;

template <typename SrcType, typename Checker>
void TestAnyViewConstructor(Checker check, TVMFFITypeIndex expected_type_index,
                            const SrcType &source) {
  AnyView v(source);
  EXPECT_EQ(v.type_index, static_cast<int32_t>(expected_type_index));
  EXPECT_EQ(v.ref_cnt, 0);
  check(&v, source);
};

int64_t FuncCall(int64_t x) { return x + 1; }

std::vector<AnyView> AnyViewArrayFactory() {
  static const char *raw_str = "Hello (raw str)";
  static std::string std_str = "World (std::string)";
  static std::string ref_str = "Hello World (Ref<Str>)";
  static Ref<Object> obj = Ref<Object>::New();
  static Ref<Func> func = Ref<Func>::New(FuncCall);
  static Ref<Str> str = Ref<Str>::New(ref_str);
  return std::vector<AnyView>{
      AnyView(nullptr),
      AnyView(1),
      AnyView(2.5),
      AnyView(reinterpret_cast<void *>(FuncCall)),
      AnyView(DLDevice{kDLCPU, 0}),
      AnyView(DLDataType{kDLInt, 32, 1}),
      AnyView(raw_str),
      AnyView(obj),
      AnyView(func),
      AnyView(std_str), // TODO: disable AnyView(std::string&&)
      AnyView(str),
  };
}

template <typename SrcType>
void TestAnyViewStringify(const SrcType &source, TVMFFITypeIndex expected_type_index,
                          const std::string &expected) {
  AnyView v(source);
  EXPECT_EQ(v.type_index, static_cast<int32_t>(expected_type_index));
  EXPECT_EQ(v.str()->c_str(), expected);
}

template <typename SrcType, typename Checker>
void TestAnyViewStringifyChecker(const SrcType &source, TVMFFITypeIndex expected_type_index,
                                 Checker check) {
  AnyView v(source);
  EXPECT_EQ(v.type_index, static_cast<int32_t>(expected_type_index));
  check(v);
}

void CheckAnyViewRefCnt(const TVMFFIAny *v) {
  if (v->type_index >= static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStaticObjectBegin)) {
    EXPECT_EQ(v->v_obj->ref_cnt, 1);
  }
}

TEST(AnyView_Constructor_0_Default, Default) {
  AnyView v;
  EXPECT_EQ(v.type_index, 0);
  EXPECT_EQ(v.ref_cnt, 0);
  EXPECT_EQ(v.v_int64, 0);
}

TEST(AnyView_Constructor_1_AnyView, Copy) {
  AnyView v1(1);
  AnyView v2(v1);
  EXPECT_EQ(v1.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v1.v_int64, 1);
  EXPECT_EQ(v2.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v2.v_int64, 1);
}

TEST(AnyView_Constructor_1_AnyView, Move) {
  AnyView v1(1);
  AnyView v2(std::move(v1));
  EXPECT_EQ(v1.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone));
  EXPECT_EQ(v1.v_int64, 0);
  EXPECT_EQ(v2.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v2.v_int64, 1);
}

TEST(AnyView_Constructor_2_Any, Copy) {
  Any v1(1);
  AnyView v2(v1);
  EXPECT_EQ(v1.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v1.v_int64, 1);
  EXPECT_EQ(v2.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v2.v_int64, 1);
}

TEST(AnyView_Constructor_2_Any, Move) {
  // ---- The following behavior is disallowed ----
  // Any v1(1);
  // AnyView v2(std::move(v1));
}

TEST(AnyView_Constructor_3_Ref, Copy) {
  Ref<Object> obj = Ref<Object>::New();
  AnyView v(obj);
  const TVMFFIAny *v_obj = v.v_obj;
  EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject));
  EXPECT_EQ(v.ref_cnt, 0);
  EXPECT_EQ(v_obj, static_cast<const void *>(obj.get()));
  EXPECT_EQ(v_obj->ref_cnt, 1);
}

TEST(AnyView_Constructor_3_Ref, Move) {
  // The following behavior is disallowed
  // Ref<Object> obj = Ref<Object>::New();
  // AnyView v(std::move(obj));
}

TEST(AnyView_Constructor_4_TypeTraits, Integer) {
  auto check = [](TVMFFIAny *v, int64_t source) -> void { EXPECT_EQ(v->v_int64, source); };
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int8_t>(1));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int16_t>(2));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int32_t>(3));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int64_t>(4));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint8_t>(1));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint16_t>(2));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint32_t>(3));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint64_t>(4));
}

TEST(AnyView_Constructor_4_TypeTraits, Float) {
  auto check = [](TVMFFIAny *v, double source) -> void { EXPECT_EQ(v->v_float64, source); };
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIFloat, static_cast<float>(3));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIFloat, static_cast<float>(4));
}

TEST(AnyView_Constructor_4_TypeTraits, Ptr) {
  int p = 4;
  auto check = [](TVMFFIAny *v, void *source) -> void { EXPECT_EQ(v->v_ptr, source); };
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFINone, static_cast<void *>(nullptr));
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIPtr, static_cast<void *>(&p));
}

TEST(AnyView_Constructor_4_TypeTraits, Device) {
  auto check = [](TVMFFIAny *v, const DLDevice &source) -> void {
    EXPECT_EQ(v->v_device.device_type, source.device_type);
    EXPECT_EQ(v->v_device.device_id, source.device_id);
  };
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIDevice, DLDevice{kDLCPU, 0});
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIDevice, DLDevice{kDLCUDA, 1});
}

TEST(AnyView_Constructor_4_TypeTraits, DataType) {
  auto check = [](TVMFFIAny *v, const DLDataType &source) -> void {
    EXPECT_EQ(v->v_dtype.code, source.code);
    EXPECT_EQ(v->v_dtype.bits, source.bits);
    EXPECT_EQ(v->v_dtype.lanes, source.lanes);
  };
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIDataType, DLDataType{kDLInt, 32, 1});
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIDataType, DLDataType{kDLUInt, 0, 0});
}

TEST(AnyView_Constructor_4_TypeTraits, RawStr) {
  auto check = [](TVMFFIAny *v, const char *source) -> void { EXPECT_EQ(v->v_str, source); };
  const char *empty = "";
  const char *hello = "hello";
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIRawStr, empty);
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIRawStr, hello);
}

TEST(AnyView_Constructor_4_TypeTraits, CharArray) {
  auto check = [](TVMFFIAny *v, const char *source) -> void { EXPECT_EQ(v->v_str, source); };
  const char empty[] = "";
  const char hello[] = "hello";
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIRawStr, empty);
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIRawStr, hello);
}

TEST(AnyView_Constructor_4_TypeTraits, StdString) {
  auto check = [](TVMFFIAny *v, const std::string &source) -> void {
    EXPECT_EQ(v->v_str, source.data());
  };
  std::string empty = "";
  std::string hello = "hello";
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIRawStr, hello);
  TestAnyViewConstructor(check, TVMFFITypeIndex::kTVMFFIRawStr, empty);
}

TEST(AnyView_Constructor_5_Object_Ptr, Object) {
  Ref<Object> obj = Ref<Object>::New();
  AnyView v(obj.get());
  EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject));
  EXPECT_EQ(v.v_obj->ref_cnt, 1);
  EXPECT_EQ(v.v_obj, static_cast<void *>(obj.get()));
}

TEST(AnyView_Constructor_5_Object_Ptr, Func) {
  Ref<Func> func = Ref<Func>::New(FuncCall);
  AnyView v(func.get());
  EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc));
  EXPECT_EQ(v.v_obj->ref_cnt, 1);
  EXPECT_EQ(v.v_ptr, static_cast<void *>(func.get()));
}

TEST(AnyView_Constructor_5_Object_Ptr, Str) {
  Ref<Str> str = Ref<Str>::New("hello");
  AnyView v(str.get());
  EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
  EXPECT_EQ(v.v_obj->ref_cnt, 1);
  EXPECT_EQ(v.v_obj, static_cast<void *>(str.get()));
}

TEST(AnyView_Converter_0_TypeTraits, Integer) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> int64_t { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt)) {
      EXPECT_EQ(convert(), 1);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `int`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_0_TypeTraits, Float) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> double { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt)) {
      EXPECT_EQ(convert(), 1.0);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFloat)) {
      EXPECT_EQ(convert(), 2.5);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `float`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_0_TypeTraits, Ptr) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> void * { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      EXPECT_EQ(convert(), nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIPtr)) {
      EXPECT_EQ(convert(), reinterpret_cast<void *>(&FuncCall));
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
      EXPECT_EQ(convert(), v.v_str);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `Ptr`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_0_TypeTraits, Device) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> DLDevice { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDevice)) {
      EXPECT_EQ(convert().device_type, kDLCPU);
      EXPECT_EQ(convert().device_id, 0);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `Device`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_0_TypeTraits, DataType) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> DLDataType { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDataType)) {
      EXPECT_EQ(convert().code, kDLInt);
      EXPECT_EQ(convert().bits, 32);
      EXPECT_EQ(convert().lanes, 1);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `dtype`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_0_TypeTraits, RawStr) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  int counter = 0;
  for (const AnyView &v : views) {
    auto convert = [&]() -> const char * { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
      counter += 1;
      EXPECT_LT(counter, 3);
      if (counter == 1) {
        EXPECT_STREQ(convert(), "Hello (raw str)");
      } else if (counter == 2) {
        EXPECT_STREQ(convert(), "World (std::string)");
      }
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      EXPECT_STREQ(convert(), "Hello World (Ref<Str>)");
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `const char *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_0_TypeTraits, RawStrToStrStar_Fail) {
  AnyView v = "Hello";
  try {
    Str *v_str = v;
    (void)v_str;
    FAIL() << "No exception thrown";
  } catch (TVMError &ex) {
    EXPECT_STREQ(ex.what(), "Cannot convert from type `const char *` to `object.Str *`");
  }
}

TEST(AnyView_Converter_0_TypeTraits, RawStrToStrStar_WrithStorage) {
  Any storage;
  AnyView v = "Hello";
  Str *v_str = v.CastWithStorage<Str *>(&storage);
  EXPECT_STREQ(v_str->c_str(), "Hello");
}

TEST(AnyView_Converter_1_Any, Any) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &view : views) {
    auto convert = [&]() -> Any { return view; };
    {
      Any ret = convert();
      EXPECT_EQ(view.ref_cnt, 0);
      if (view.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
        Str *str = ret;
        EXPECT_EQ(ret.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
        EXPECT_STREQ(str->c_str(), view.v_str);
        EXPECT_EQ(ret.ref_cnt, 0);
      } else {
        EXPECT_EQ(ret.type_index, view.type_index);
        EXPECT_EQ(ret.ref_cnt, 0);
        EXPECT_EQ(ret.v_obj, view.v_obj);
      }
    }
    CheckAnyViewRefCnt(&view);
  }
}

TEST(AnyView_Converter_2_Ref, Object) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> Ref<Object> { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Ref<Object> ret = convert();
      EXPECT_EQ(ret.get(), nullptr);
    } else if (v.type_index >= static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStaticObjectBegin)) {
      Ref<Object> ret = convert();
      EXPECT_EQ(ret.get(), static_cast<void *>(v.v_obj));
      EXPECT_EQ(v.v_obj->ref_cnt, 2);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Object`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_2_Ref, Func) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> Ref<Func> { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Ref<Func> ret = convert();
      EXPECT_EQ(ret.get(), nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc)) {
      Ref<Func> ret = convert();
      EXPECT_EQ(ret.get(), static_cast<void *>(v.v_obj));
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Func`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_2_Ref, Str) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> Ref<Str> { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Ref<Str> ret = convert();
      EXPECT_EQ(ret.get(), nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      Ref<Str> ret = convert();
      EXPECT_STREQ(ret->c_str(), "Hello World (Ref<Str>)");
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
      Ref<Str> ret = convert();
      EXPECT_EQ(reinterpret_cast<TVMFFIStr *>(ret.get())->ref_cnt, 1);
      EXPECT_STREQ(ret->c_str(), v.v_str);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Str`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_3_Object_Ptr, Object) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> Object * { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Object *ret = convert();
      EXPECT_EQ(ret, nullptr);
    } else if (v.type_index >= static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStaticObjectBegin)) {
      Object *ret = convert();
      EXPECT_EQ(ret, static_cast<void *>(v.v_obj));
      EXPECT_EQ(v.v_obj->ref_cnt, 1);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Object *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_3_Object_Ptr, Func) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> Func * { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Func *ret = convert();
      EXPECT_EQ(ret, nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc)) {
      Func *ret = convert();
      EXPECT_EQ(ret, reinterpret_cast<Func *>(v.v_ptr));
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Func *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Converter_3_Object_Ptr, Str) {
  std::vector<AnyView> views = AnyViewArrayFactory();
  for (const AnyView &v : views) {
    auto convert = [&]() -> Str * { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Str *ret = convert();
      EXPECT_EQ(ret, nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      Str *ret = convert();
      EXPECT_STREQ(ret->c_str(), "Hello World (Ref<Str>)");
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError &ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Str *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyViewRefCnt(&v);
  }
}

TEST(AnyView_Stringify, Integer) {
  TestAnyViewStringify<int8_t>(-13, TVMFFITypeIndex::kTVMFFIInt, "-13");
  TestAnyViewStringify<int16_t>(-5, TVMFFITypeIndex::kTVMFFIInt, "-5");
  TestAnyViewStringify<int32_t>(0, TVMFFITypeIndex::kTVMFFIInt, "0");
  TestAnyViewStringify<int64_t>(1, TVMFFITypeIndex::kTVMFFIInt, "1");
}

TEST(AnyView_Stringify, Float) {
  auto check = [](const AnyView &v) -> void {
    std::string str = v.str()->c_str();
    double f_str = std::stod(str);
    double f_src = v.v_float64;
    EXPECT_NEAR(f_src, f_str, 1e-5);
  };
  TestAnyViewStringifyChecker<float>(float(-3.14), TVMFFITypeIndex::kTVMFFIFloat, check);
  TestAnyViewStringifyChecker<double>(0.0, TVMFFITypeIndex::kTVMFFIFloat, check);
}

TEST(AnyView_Stringify, Ptr) {
  auto check = [](const AnyView &v) -> void {
    std::string str = v.str()->c_str();
    EXPECT_GT(str.size(), 2);
  };
  TestAnyViewStringify<void *>(nullptr, TVMFFITypeIndex::kTVMFFINone, "None");
  TestAnyViewStringifyChecker<void *>(reinterpret_cast<void *>(FuncCall),
                                      TVMFFITypeIndex::kTVMFFIPtr, check);
}

TEST(AnyView_Stringify, Device) {
  TestAnyViewStringify<DLDevice>(DLDevice{kDLCPU, 0}, TVMFFITypeIndex::kTVMFFIDevice, "cpu:0");
  TestAnyViewStringify<DLDevice>(DLDevice{kDLCUDA, 1}, TVMFFITypeIndex::kTVMFFIDevice, "cuda:1");
}

TEST(AnyView_Stringify, DataType) {
  TestAnyViewStringify<DLDataType>(DLDataType{kDLInt, 32, 1}, TVMFFITypeIndex::kTVMFFIDataType,
                                   "int32");
  TestAnyViewStringify<DLDataType>(DLDataType{kDLUInt, 1, 1}, TVMFFITypeIndex::kTVMFFIDataType,
                                   "bool");
  TestAnyViewStringify<DLDataType>(DLDataType{kDLOpaqueHandle, 0, 0},
                                   TVMFFITypeIndex::kTVMFFIDataType, "void");
  TestAnyViewStringify<DLDataType>(DLDataType{kDLFloat, 8, 4}, TVMFFITypeIndex::kTVMFFIDataType,
                                   "float8x4");
}

TEST(AnyView_Stringify, RawStr) {
  TestAnyViewStringify<const char *>("Hello", TVMFFITypeIndex::kTVMFFIRawStr, "\"Hello\"");
  TestAnyViewStringify<char[6]>("Hello", TVMFFITypeIndex::kTVMFFIRawStr, "\"Hello\"");
  TestAnyViewStringify<const char[6]>("Hello", TVMFFITypeIndex::kTVMFFIRawStr, "\"Hello\"");
}

TEST(AnyView_Stringify, Object) {
  auto check = [](const AnyView &v) -> void {
    std::string expected_prefix = "object.Object@0";
    int n = static_cast<int>(expected_prefix.size());
    std::string str = v.str()->c_str();
    EXPECT_GT(str.size(), n);
    EXPECT_EQ(str.substr(0, n), expected_prefix);
  };
  TestAnyViewStringifyChecker<Ref<Object>>(Ref<Object>::New(), TVMFFITypeIndex::kTVMFFIObject,
                                           check);
}

TEST(AnyView_Stringify, Func) {
  auto check = [](const AnyView &v) -> void {
    std::string expected_prefix = "object.Func@0";
    int n = static_cast<int>(expected_prefix.size());
    std::string str = v.str()->c_str();
    EXPECT_GT(str.size(), n);
    EXPECT_EQ(str.substr(0, n), expected_prefix);
  };
  TestAnyViewStringifyChecker<Ref<Func>>(Ref<Func>::New(FuncCall), TVMFFITypeIndex::kTVMFFIFunc,
                                         check);
}

TEST(AnyView_Stringify, Str) {
  auto check = [](const AnyView &v) -> void {
    std::string str = v.str()->c_str();
    EXPECT_EQ(str, "\"Hello World\"");
  };
  TestAnyViewStringifyChecker<Ref<Str>>(Ref<Str>::New("Hello World"), TVMFFITypeIndex::kTVMFFIStr,
                                        check);
}

} // namespace
