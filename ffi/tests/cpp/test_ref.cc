#include <gtest/gtest.h>
#include <tvm/ffi/ffi.hpp>
#include <unordered_map>

namespace {
using namespace tvm::ffi;
using tvm::ffi::details::StrPad;
using tvm::ffi::details::StrStd;

using ObjDeleter = void (*)(void *);

struct AllocRecorder {
  std::unordered_map<void *, ObjDeleter> deleters;

  void Alloc(void *ptr) {
    deleters[ptr] = reinterpret_cast<TVMFFIAny *>(ptr)->deleter;
    reinterpret_cast<TVMFFIAny *>(ptr)->deleter = AllocRecorder::Deleter;
  }

  void Delete(void *ptr) {
    ASSERT_EQ(deleters.count(ptr), 1);
    ObjDeleter d = this->deleters[ptr];
    d(ptr);
    deleters.erase(ptr);
  }

  bool IsDeletedImpl(void *ptr) { return deleters.count(ptr) == 0; }

  static void Deleter(void *ptr) { AllocRecorder::Global()->Delete(ptr); }

  static bool IsDeleted(void *ptr) { return AllocRecorder::Global()->IsDeletedImpl(ptr); }

  static AllocRecorder *Global() {
    static AllocRecorder inst;
    return &inst;
  }
};

template <typename ObjectType>
struct TestAllocator {
  using Allocator = typename ::tvm::ffi::GetAllocator<ObjectType>::Type;

  template <typename... Args>
  TVM_FFI_INLINE static ObjectType *New(Args &&...args) {
    ObjectType *ret = Allocator::New(std::forward<Args>(args)...);
    AllocRecorder::Global()->Alloc(ret);
    return ret;
  }

  template <typename PadType, typename... Args>
  TVM_FFI_INLINE static ObjectType *NewWithPad(size_t pad_size, Args &&...args) {
    ObjectType *ret =
        Allocator::template NewWithPad<PadType>(pad_size, std::forward<Args>(args)...);
    AllocRecorder::Global()->Alloc(ret);
    return ret;
  }
};

int64_t FuncCall(int64_t x) { return x + 1; }

int32_t GetRefCount(void *obj) { return reinterpret_cast<TVMFFIAny *>(obj)->ref_cnt; }

int32_t GetTypeIndex(void *obj) { return reinterpret_cast<TVMFFIAny *>(obj)->type_index; }

ObjDeleter GetDeleter(void *obj) { return reinterpret_cast<TVMFFIAny *>(obj)->deleter; }

TEST(Ref_Constructor_0_Default, Default) {
  Ref<Object> ref;
  EXPECT_EQ(ref.get(), nullptr);
}

TEST(Ref_Constructor_1_Ptr, SameType) {
  Object *obj = TestAllocator<Object>::New();
  {
    Ref<Object> ref(obj);
    EXPECT_EQ(ref.get(), obj);
    EXPECT_EQ(GetRefCount(obj), 1);
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_1_Ptr, SubType) {
  Str *obj = TestAllocator<Str>::New("Hello world");
  {
    Ref<Object> ref(obj);
    EXPECT_EQ(ref.get(), static_cast<void *>(obj));
    EXPECT_EQ(GetRefCount(obj), 1);
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_2_Ref, SameType_Copy) {
  Object *obj = TestAllocator<Object>::New();
  {
    Ref<Object> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      Ref<Object> ref2(ref1);
      EXPECT_EQ(GetRefCount(obj), 2);
    }
    EXPECT_EQ(GetRefCount(ref1.get()), 1);
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_2_Ref, SameType_Move) {
  Object *obj = TestAllocator<Object>::New();
  {
    Ref<Object> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      Ref<Object> ref2(std::move(ref1));
      EXPECT_EQ(GetRefCount(obj), 1);
      EXPECT_EQ(ref1.get(), nullptr);
    }
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_2_Ref, SubType_Copy) {
  Str *obj = TestAllocator<Str>::New("Hello world");
  {
    Ref<Str> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      Ref<Object> ref2(ref1);
      EXPECT_EQ(GetRefCount(obj), 2);
    }
    EXPECT_EQ(GetRefCount(obj), 1);
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_2_Ref, SubType_Move) {
  Str *obj = TestAllocator<Str>::New("Hello world");
  {
    Ref<Str> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      Ref<Object> ref2(std::move(ref1));
      EXPECT_EQ(GetRefCount(obj), 1);
      EXPECT_EQ(ref1.get(), nullptr);
    }
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_3_AnyView, Copy) {
  Object *obj = TestAllocator<Object>::New();
  {
    Ref<Object> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      AnyView view(ref1);
      Ref<Object> ref2(view);
      EXPECT_EQ(GetRefCount(obj), 2);
    }
    EXPECT_EQ(GetRefCount(obj), 1);
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_3_AnyView, Move) {
  Object *obj = TestAllocator<Object>::New();
  {
    Ref<Object> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      AnyView view(ref1);
      EXPECT_EQ(GetRefCount(obj), 1);
      {
        Ref<Object> ref2(std::move(view));
        EXPECT_EQ(GetRefCount(obj), 2);
        EXPECT_EQ(view.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone));
        EXPECT_EQ(view.ref_cnt, 0);
        EXPECT_EQ(view.v_int64, 0);
      }
      EXPECT_EQ(GetRefCount(obj), 1);
    }
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_4_Any, Copy) {
  Object *obj = TestAllocator<Object>::New();
  {
    Ref<Object> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      Any any(ref1);
      EXPECT_EQ(GetRefCount(obj), 2);
      Ref<Object> ref2(any);
      EXPECT_EQ(GetRefCount(obj), 3);
    }
    EXPECT_EQ(GetRefCount(obj), 1);
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_Constructor_4_Any, Move) {
  Object *obj = TestAllocator<Object>::New();
  {
    Ref<Object> ref1(obj);
    EXPECT_EQ(GetRefCount(obj), 1);
    {
      Any any(ref1);
      EXPECT_EQ(GetRefCount(obj), 2);
      {
        Ref<Object> ref2(std::move(any));
        EXPECT_EQ(GetRefCount(obj), 2);
        EXPECT_EQ(any.type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone));
        EXPECT_EQ(any.ref_cnt, 0);
        EXPECT_EQ(any.v_int64, 0);
      }
      EXPECT_EQ(GetRefCount(obj), 1);
    }
  }
  EXPECT_TRUE(AllocRecorder::IsDeleted(obj));
}

TEST(Ref_New, Object) {
  Ref<Object> ref = Ref<Object>::New();
  EXPECT_EQ(GetTypeIndex(ref.get()), static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject));
  EXPECT_EQ(GetRefCount(ref.get()), 1);
}

TEST(Ref_New, Func) {
  Ref<Func> ref = Ref<Func>::New(FuncCall);
  EXPECT_EQ(GetTypeIndex(ref.get()), static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc));
  EXPECT_EQ(GetRefCount(ref.get()), 1);
}

TEST(Ref_New, RawStr) {
  const char *str = "Hello world";
  Ref<Str> ref = Ref<Str>::New(str);
  EXPECT_EQ(GetTypeIndex(ref.get()), static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
  EXPECT_EQ(GetRefCount(ref.get()), 1);
  EXPECT_EQ(GetDeleter(ref.get()), DefaultObjectAllocator<StrPad>::Deleter);
}

TEST(Ref_New, CharArray) {
  const char str[18] = "Hello world";
  Ref<Str> ref = Ref<Str>::New(str);
  EXPECT_EQ(GetTypeIndex(ref.get()), static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
  EXPECT_EQ(GetRefCount(ref.get()), 1);
  EXPECT_EQ(GetDeleter(ref.get()), DefaultObjectAllocator<StrPad>::Deleter);
  EXPECT_EQ(ref->size(), 17);
}

TEST(Ref_New, StdString_Copy) {
  std::string str = "Hello world";
  Ref<Str> ref = Ref<Str>::New(str);
  EXPECT_EQ(GetTypeIndex(ref.get()), static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
  EXPECT_EQ(GetRefCount(ref.get()), 1);
  EXPECT_EQ(GetDeleter(ref.get()), DefaultObjectAllocator<StrPad>::Deleter);
}

TEST(Ref_New, StdString_Move) {
  std::string str = "Hello world";
  Ref<Str> ref = Ref<Str>::New(std::move(str));
  EXPECT_EQ(GetTypeIndex(ref.get()), static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
  EXPECT_EQ(GetRefCount(ref.get()), 1);
  EXPECT_EQ(GetDeleter(ref.get()), DefaultObjectAllocator<StrStd>::Deleter);
}

TEST(Ref_Stringify, Object) {
  std::string str = Ref<Object>::New().str()->c_str();
  std::string expected_prefix = "object.Object@0";
  EXPECT_GT(str.size(), expected_prefix.size());
  EXPECT_EQ(str.substr(0, expected_prefix.size()), expected_prefix);
}

TEST(Ref_Stringify, Func) {
  std::string str = Ref<Func>::New(FuncCall).str()->c_str();
  std::string expected_prefix = "object.Func@0";
  EXPECT_GT(str.size(), expected_prefix.size());
  EXPECT_EQ(str.substr(0, expected_prefix.size()), expected_prefix);
}

TEST(Ref_Stringify, Str) {
  std::string str = Ref<Str>::New("Hello world").str()->c_str();
  EXPECT_EQ(str, "\"Hello world\"");
}

TEST(Ref_Misc, MoveToRaw) {
  TVMFFIAny *str = reinterpret_cast<TVMFFIAny *>(Ref<Str>::New("Hello world").MoveToRawObjPtr());
  EXPECT_EQ(GetTypeIndex(str), static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
  EXPECT_EQ(GetRefCount(str), 1);
  EXPECT_EQ(GetDeleter(str), DefaultObjectAllocator<StrPad>::Deleter);
  str->deleter(str);
}

} // namespace
