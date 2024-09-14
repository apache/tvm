#include <gtest/gtest.h>

#include <tvm/ffi/ffi.hpp>

namespace {
using namespace tvm::ffi;

template <typename SrcType, typename Checker>
void TestAnyConstructor(Checker check, TVMFFITypeIndex expected_type_index, const SrcType& source) {
  Any v(source);
  EXPECT_EQ(v.type_index, static_cast<int32_t>(expected_type_index));
  EXPECT_EQ(v.ref_cnt, 0);
  check(&v, source);
};

int64_t FuncCall(int64_t x) { return x + 1; }

std::vector<Any> AnyArrayFactory() {
  return std::vector<Any>{
      Any(nullptr),
      Any(1),
      Any(2.5),
      Any(reinterpret_cast<void*>(FuncCall)),
      Any(DLDevice{kDLCPU, 0}),
      Any(DLDataType{kDLInt, 32, 1}),
      Any("Hello (raw str)"),
      Any(Ref<Object>::New()),
      Any(Ref<Func>::New(FuncCall)),
      Any(std::string("World (std::string)")),
      Any(Ref<Str>::New("Hello World (Ref<Str>)")),
  };
}

template <typename SrcType>
void TestAnyStringify(const SrcType& source, TVMFFITypeIndex expected_type_index,
                      const std::string& expected) {
  Any v(source);
  EXPECT_EQ(v.type_index, static_cast<int32_t>(expected_type_index));
  EXPECT_EQ(v.str()->c_str(), expected);
}

template <typename SrcType, typename Checker>
void TestAnyStringifyChecker(const SrcType& source, TVMFFITypeIndex expected_type_index,
                             Checker check) {
  Any v(source);
  EXPECT_EQ(v.type_index, static_cast<int32_t>(expected_type_index));
  check(v);
}

void CheckAnyRefCnt(const TVMFFIAny* v) {
  if (v->type_index >= static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStaticObjectBegin)) {
    EXPECT_EQ(v->v_obj->ref_cnt, 1);
  }
}

TEST(Any_Constructor_0_Default, Default) {
  Any v;
  EXPECT_EQ(v.type_index, 0);
  EXPECT_EQ(v.ref_cnt, 0);
  EXPECT_EQ(v.v_int64, 0);
}

TEST(Any_Constructor_1_Any, Copy) {
  Any v1(1);
  Any v2(v1);
  EXPECT_EQ(v1.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v1.v_int64, 1);
  EXPECT_EQ(v2.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v2.v_int64, 1);
}

TEST(Any_Constructor_1_Any, Move) {
  Any v1(1);
  Any v2(std::move(v1));
  EXPECT_EQ(v1.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone));
  EXPECT_EQ(v1.v_int64, 0);
  EXPECT_EQ(v2.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v2.v_int64, 1);
}

TEST(Any_Constructor_2_AnyView, Copy) {
  AnyView v1(1);
  Any v2(v1);
  EXPECT_EQ(v1.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v1.v_int64, 1);
  EXPECT_EQ(v2.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v2.v_int64, 1);
}

TEST(Any_Constructor_2_AnyView, Move) {
  AnyView v1(1);
  Any v2(std::move(v1));
  EXPECT_EQ(v1.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone));
  EXPECT_EQ(v1.v_int64, 0);
  EXPECT_EQ(v2.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt));
  EXPECT_EQ(v2.v_int64, 1);
}

TEST(Any_Constructor_3_Ref, Copy) {
  Ref<Object> obj = Ref<Object>::New();
  Any v(obj);
  const TVMFFIAny* v_obj = v.v_obj;
  EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject));
  EXPECT_EQ(v.ref_cnt, 0);
  EXPECT_EQ(v_obj, static_cast<const void*>(obj.get()));
  EXPECT_EQ(v_obj->ref_cnt, 2);
}

TEST(Any_Constructor_3_Ref, Move) {
  Ref<Object> obj = Ref<Object>::New();
  Any v(std::move(obj));
  const TVMFFIAny* v_obj = v.v_obj;
  EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject));
  EXPECT_EQ(v.ref_cnt, 0);
  EXPECT_EQ(v_obj->ref_cnt, 1);
  EXPECT_EQ(obj.get(), nullptr);
}

TEST(Any_Constructor_4_TypeTraits, Integer) {
  auto check = [](TVMFFIAny* v, int64_t source) -> void { EXPECT_EQ(v->v_int64, source); };
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int8_t>(1));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int16_t>(2));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int32_t>(3));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<int64_t>(4));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint8_t>(1));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint16_t>(2));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint32_t>(3));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIInt, static_cast<uint64_t>(4));
}

TEST(Any_Constructor_4_TypeTraits, Float) {
  auto check = [](TVMFFIAny* v, double source) -> void { EXPECT_EQ(v->v_float64, source); };
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIFloat, static_cast<float>(3));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIFloat, static_cast<float>(4));
}

TEST(Any_Constructor_4_TypeTraits, Ptr) {
  int p = 4;
  auto check = [](TVMFFIAny* v, void* source) -> void { EXPECT_EQ(v->v_ptr, source); };
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFINone, static_cast<void*>(nullptr));
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIPtr, static_cast<void*>(&p));
}

TEST(Any_Constructor_4_TypeTraits, Device) {
  auto check = [](TVMFFIAny* v, const DLDevice& source) -> void {
    EXPECT_EQ(v->v_device.device_type, source.device_type);
    EXPECT_EQ(v->v_device.device_id, source.device_id);
  };
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIDevice, DLDevice{kDLCPU, 0});
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIDevice, DLDevice{kDLCUDA, 1});
}

TEST(Any_Constructor_4_TypeTraits, DataType) {
  auto check = [](TVMFFIAny* v, const DLDataType& source) -> void {
    EXPECT_EQ(v->v_dtype.code, source.code);
    EXPECT_EQ(v->v_dtype.bits, source.bits);
    EXPECT_EQ(v->v_dtype.lanes, source.lanes);
  };
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIDataType, DLDataType{kDLInt, 32, 1});
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIDataType, DLDataType{kDLUInt, 0, 0});
}

TEST(Any_Constructor_4_TypeTraits, RawStr) {
  auto check = [](TVMFFIAny* v, const char* source) -> void {
    Str* str = static_cast<Str*>(v->v_ptr);
    EXPECT_STREQ(str->c_str(), source);
  };
  const char* empty = "";
  const char* hello = "hello";
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIStr, empty);
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIStr, hello);
}

TEST(Any_Constructor_4_TypeTraits, CharArray) {
  auto check = [](TVMFFIAny* v, const char* source) -> void {
    Str* str = static_cast<Str*>(v->v_ptr);
    EXPECT_STREQ(str->c_str(), source);
  };
  const char empty[] = "";
  const char hello[] = "hello";
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIStr, empty);
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIStr, hello);
}

TEST(Any_Constructor_4_TypeTraits, StdString) {
  auto check = [](TVMFFIAny* v, const std::string& source) -> void {
    Str* str = static_cast<Str*>(v->v_ptr);
    EXPECT_EQ(str->c_str(), source);
  };
  std::string empty = "";
  std::string hello = "hello";
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIStr, hello);
  TestAnyConstructor(check, TVMFFITypeIndex::kTVMFFIStr, empty);
}

TEST(Any_Constructor_5_Object_Ptr, Object) {
  Ref<Object> obj = Ref<Object>::New();
  TVMFFIAny* ptr = reinterpret_cast<TVMFFIAny*>(obj.get());
  {
    Any v(obj.get());
    EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject));
    EXPECT_EQ(v.v_obj->ref_cnt, 2);
    EXPECT_EQ(v.v_obj, ptr);
  }
  EXPECT_EQ(ptr->ref_cnt, 1);
}

TEST(Any_Constructor_5_Object_Ptr, Func) {
  Ref<Func> func = Ref<Func>::New(FuncCall);
  TVMFFIAny* ptr = reinterpret_cast<TVMFFIAny*>(func.get());
  {
    Any v(func.get());
    EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc));
    EXPECT_EQ(v.v_obj->ref_cnt, 2);
    EXPECT_EQ(v.v_obj, ptr);
  }
  EXPECT_EQ(ptr->ref_cnt, 1);
}

TEST(Any_Constructor_5_Object_Ptr, Str) {
  Ref<Str> str = Ref<Str>::New("hello");
  TVMFFIAny* ptr = reinterpret_cast<TVMFFIAny*>(str.get());
  {
    Any v(str.get());
    EXPECT_EQ(v.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
    EXPECT_EQ(v.v_obj->ref_cnt, 2);
    EXPECT_EQ(v.v_obj, ptr);
  }
  EXPECT_EQ(ptr->ref_cnt, 1);
}

TEST(Any_Converter_0_TypeTraits, Integer) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> int64_t { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt)) {
      EXPECT_EQ(convert(), 1);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `int`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_0_TypeTraits, Float) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> double { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt)) {
      EXPECT_EQ(convert(), 1.0);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFloat)) {
      EXPECT_EQ(convert(), 2.5);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `float`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_0_TypeTraits, Ptr) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> void* { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      EXPECT_EQ(convert(), nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIPtr)) {
      EXPECT_EQ(convert(), reinterpret_cast<void*>(&FuncCall));
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
      EXPECT_EQ(convert(), v.v_str);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `Ptr`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_0_TypeTraits, Device) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> DLDevice { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDevice)) {
      EXPECT_EQ(convert().device_type, kDLCPU);
      EXPECT_EQ(convert().device_id, 0);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `Device`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_0_TypeTraits, DataType) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> DLDataType { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDataType)) {
      EXPECT_EQ(convert().code, kDLInt);
      EXPECT_EQ(convert().bits, 32);
      EXPECT_EQ(convert().lanes, 1);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index) << "` to `dtype`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_0_TypeTraits, RawStr) {
  std::vector<Any> vs = AnyArrayFactory();
  int counter = 0;
  for (const Any& v : vs) {
    auto convert = [&]() -> const char* { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      ++counter;
      EXPECT_LE(counter, 3);
      if (counter == 1) {
        EXPECT_STREQ(convert(), "Hello (raw str)");
      } else if (counter == 2) {
        EXPECT_STREQ(convert(), "World (std::string)");
      } else if (counter == 3) {
        EXPECT_STREQ(convert(), "Hello World (Ref<Str>)");
      }
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `const char *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_1_AnyView, Any) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> AnyView { return v; };
    {
      AnyView ret = convert();
      EXPECT_EQ(ret.type_index, v.type_index);
      EXPECT_EQ(ret.ref_cnt, 0);
      EXPECT_EQ(v.ref_cnt, 0);
      EXPECT_EQ(ret.v_obj, v.v_obj);
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_2_Ref, Object) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> Ref<Object> { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Ref<Object> ret = convert();
      EXPECT_EQ(ret.get(), nullptr);
    } else if (v.type_index >= static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStaticObjectBegin)) {
      Ref<Object> ret = convert();
      EXPECT_EQ(ret.get(), static_cast<void*>(v.v_obj));
      EXPECT_EQ(v.v_obj->ref_cnt, 2);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Object`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_2_Ref, Func) {
  std::vector<Any> views = AnyArrayFactory();
  for (const Any& v : views) {
    auto convert = [&]() -> Ref<Func> { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Ref<Func> ret = convert();
      EXPECT_EQ(ret.get(), nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc)) {
      Ref<Func> ret = convert();
      EXPECT_EQ(ret.get(), static_cast<void*>(v.v_obj));
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Func`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_2_Ref, Str) {
  int counter = 0;
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> Ref<Str> { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Ref<Str> ret = convert();
      EXPECT_EQ(ret.get(), nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      Ref<Str> ret = convert();
      ++counter;
      EXPECT_LE(counter, 3);
      if (counter == 1) {
        EXPECT_STREQ(ret->c_str(), "Hello (raw str)");
      } else if (counter == 2) {
        EXPECT_STREQ(ret->c_str(), "World (std::string)");
      } else {
        EXPECT_STREQ(ret->c_str(), "Hello World (Ref<Str>)");
      }
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Str`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_3_Object_Ptr, Object) {
  std::vector<Any> vs = AnyArrayFactory();
  for (const Any& v : vs) {
    auto convert = [&]() -> Object* { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Object* ret = convert();
      EXPECT_EQ(ret, nullptr);
    } else if (v.type_index >= static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStaticObjectBegin)) {
      Object* ret = convert();
      EXPECT_EQ(ret, static_cast<void*>(v.v_obj));
      EXPECT_EQ(v.v_obj->ref_cnt, 1);
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Object *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_3_Object_Ptr, Func) {
  std::vector<Any> views = AnyArrayFactory();
  for (const Any& v : views) {
    auto convert = [&]() -> Func* { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Func* ret = convert();
      EXPECT_EQ(ret, nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc)) {
      Func* ret = convert();
      EXPECT_EQ(ret, reinterpret_cast<Func*>(v.v_ptr));
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Func *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Converter_3_Object_Ptr, Str) {
  std::vector<Any> vs = AnyArrayFactory();
  int counter = 0;
  for (const Any& v : vs) {
    auto convert = [&]() -> Str* { return v; };
    if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)) {
      Str* ret = convert();
      EXPECT_EQ(ret, nullptr);
    } else if (v.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr)) {
      Str* ret = convert();
      ++counter;
      EXPECT_LE(counter, 3);
      if (counter == 1) {
        EXPECT_STREQ(ret->c_str(), "Hello (raw str)");
      } else if (counter == 2) {
        EXPECT_STREQ(ret->c_str(), "World (std::string)");
      } else {
        EXPECT_STREQ(ret->c_str(), "Hello World (Ref<Str>)");
      }
    } else {
      try {
        convert();
        FAIL() << "No exception thrown";
      } catch (TVMError& ex) {
        std::ostringstream os;
        os << "Cannot convert from type `" << TypeIndex2TypeKey(v.type_index)
           << "` to `object.Str *`";
        EXPECT_EQ(ex.what(), os.str());
      }
    }
    CheckAnyRefCnt(&v);
  }
}

TEST(Any_Stringify, Integer) {
  TestAnyStringify<int8_t>(-13, TVMFFITypeIndex::kTVMFFIInt, "-13");
  TestAnyStringify<int16_t>(-5, TVMFFITypeIndex::kTVMFFIInt, "-5");
  TestAnyStringify<int32_t>(0, TVMFFITypeIndex::kTVMFFIInt, "0");
  TestAnyStringify<int64_t>(1, TVMFFITypeIndex::kTVMFFIInt, "1");
}

TEST(Any_Stringify, Float) {
  auto check = [](const Any& v) -> void {
    std::string str = v.str()->c_str();
    double f_str = std::stod(str);
    double f_src = v.v_float64;
    EXPECT_NEAR(f_src, f_str, 1e-5);
  };
  TestAnyStringifyChecker<float>(float(-3.14), TVMFFITypeIndex::kTVMFFIFloat, check);
  TestAnyStringifyChecker<double>(0.0, TVMFFITypeIndex::kTVMFFIFloat, check);
}

TEST(Any_Stringify, Ptr) {
  auto check = [](const Any& v) -> void {
    std::string str = v.str()->c_str();
    EXPECT_GT(str.size(), 2);
  };
  TestAnyStringify<void*>(nullptr, TVMFFITypeIndex::kTVMFFINone, "None");
  TestAnyStringifyChecker<void*>(reinterpret_cast<void*>(FuncCall), TVMFFITypeIndex::kTVMFFIPtr,
                                 check);
}

TEST(Any_Stringify, Device) {
  TestAnyStringify<DLDevice>(DLDevice{kDLCPU, 0}, TVMFFITypeIndex::kTVMFFIDevice, "cpu:0");
  TestAnyStringify<DLDevice>(DLDevice{kDLCUDA, 1}, TVMFFITypeIndex::kTVMFFIDevice, "cuda:1");
}

TEST(Any_Stringify, DataType) {
  TestAnyStringify<DLDataType>(DLDataType{kDLInt, 32, 1}, TVMFFITypeIndex::kTVMFFIDataType,
                               "int32");
  TestAnyStringify<DLDataType>(DLDataType{kDLUInt, 1, 1}, TVMFFITypeIndex::kTVMFFIDataType, "bool");
  TestAnyStringify<DLDataType>(DLDataType{kDLOpaqueHandle, 0, 0}, TVMFFITypeIndex::kTVMFFIDataType,
                               "void");
  TestAnyStringify<DLDataType>(DLDataType{kDLFloat, 8, 4}, TVMFFITypeIndex::kTVMFFIDataType,
                               "float8x4");
}

TEST(Any_Stringify, RawStr) {
  TestAnyStringify<const char*>("Hello", TVMFFITypeIndex::kTVMFFIStr, "\"Hello\"");
  TestAnyStringify<char[6]>("Hello", TVMFFITypeIndex::kTVMFFIStr, "\"Hello\"");
  TestAnyStringify<const char[6]>("Hello", TVMFFITypeIndex::kTVMFFIStr, "\"Hello\"");
}

TEST(Any_Stringify, Object) {
  auto check = [](const Any& v) -> void {
    std::string expected_prefix = "object.Object@0";
    int n = static_cast<int>(expected_prefix.size());
    std::string str = v.str()->c_str();
    EXPECT_GT(str.size(), n);
    EXPECT_EQ(str.substr(0, n), expected_prefix);
  };
  TestAnyStringifyChecker<Ref<Object>>(Ref<Object>::New(), TVMFFITypeIndex::kTVMFFIObject, check);
}

TEST(Any_Stringify, Func) {
  auto check = [](const Any& v) -> void {
    std::string expected_prefix = "object.Func@0";
    int n = static_cast<int>(expected_prefix.size());
    std::string str = v.str()->c_str();
    EXPECT_GT(str.size(), n);
    EXPECT_EQ(str.substr(0, n), expected_prefix);
  };
  TestAnyStringifyChecker<Ref<Func>>(Ref<Func>::New(FuncCall), TVMFFITypeIndex::kTVMFFIFunc, check);
}

TEST(Any_Stringify, Str) {
  auto check = [](const Any& v) -> void {
    std::string str = v.str()->c_str();
    EXPECT_EQ(str, "\"Hello World\"");
  };
  TestAnyStringifyChecker<Ref<Str>>(Ref<Str>::New("Hello World"), TVMFFITypeIndex::kTVMFFIStr,
                                    check);
}

}  // namespace
