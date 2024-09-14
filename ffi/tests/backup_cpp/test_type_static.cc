#include <gtest/gtest.h>

#include <tvm/ffi/ffi.hpp>

namespace {
using namespace tvm::ffi;

struct SubType : public Object {
  int data;
  explicit SubType(int data) : data(data) {
    if (data == 1) {
      throw std::runtime_error("New Error");
    }
  }
};

int64_t FuncCall(int64_t x) { return x + 1; }

void CheckAncestor(int32_t num, const int32_t* ancestors, std::vector<int32_t> expected) {
  EXPECT_EQ(num, expected.size());
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(ancestors[i], expected[i]);
  }
}

static_assert(IsObject<Object>, "IsObject<Object> == true");
static_assert(IsObject<Func>, "IsObject<Func> == true");
static_assert(IsObject<Str>, "IsObject<Str> == true");

TEST(StaticTypeInfo, Object) {
  EXPECT_EQ(Object::_type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject));
  EXPECT_STRCASEEQ(Object::_type_key, "object.Object");
  EXPECT_EQ(Object::_type_depth, 0);
  CheckAncestor(Object::_type_depth, Object::_type_ancestors.data(), {});
}

TEST(StaticTypeInfo, Func) {
  EXPECT_EQ(Func::_type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFunc));
  EXPECT_STRCASEEQ(Func::_type_key, "object.Func");
  EXPECT_EQ(Func::_type_depth, 1);
  CheckAncestor(Func::_type_depth, Func::_type_ancestors.data(),
                {static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject)});
}

TEST(StaticTypeInfo, Str) {
  EXPECT_EQ(Str::_type_index, static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr));
  EXPECT_STRCASEEQ(Str::_type_key, "object.Str");
  EXPECT_EQ(Str::_type_depth, 1);
  CheckAncestor(Str::_type_depth, Str::_type_ancestors.data(),
                {static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject)});
}

TEST(StaticTypeInheritance, None) {
  Ref<Object> obj;
  // FIXME: The lines below are going to segfault
  // EXPECT_STREQ(obj->GetTypeKey(), "None");
  // EXPECT_FALSE(obj->IsInstance<Object>());
  // EXPECT_FALSE(obj->IsInstance<Func>());
  // EXPECT_FALSE(obj->IsInstance<Str>());
}

TEST(StaticTypeInheritance, Object) {
  Ref<Object> obj = Ref<Object>::New();
  EXPECT_STREQ(obj->GetTypeKey(), "object.Object");
  EXPECT_TRUE(obj->IsInstance<Object>());
  EXPECT_FALSE(obj->IsInstance<Func>());
  EXPECT_FALSE(obj->IsInstance<Str>());
}

TEST(StaticTypeInheritance, Func_0) {
  Ref<Func> obj = Ref<Func>::New(FuncCall);
  EXPECT_STREQ(obj->GetTypeKey(), "object.Func");
  EXPECT_TRUE(obj->IsInstance<Object>());
  EXPECT_TRUE(obj->IsInstance<Func>());
  EXPECT_FALSE(obj->IsInstance<Str>());
}

TEST(StaticTypeInheritance, Func_1) {
  Ref<Object> obj = Ref<Func>::New(FuncCall);
  EXPECT_STREQ(obj->GetTypeKey(), "object.Func");
  EXPECT_TRUE(obj->IsInstance<Object>());
  EXPECT_TRUE(obj->IsInstance<Func>());
  EXPECT_FALSE(obj->IsInstance<Str>());
}

TEST(StaticTypeInheritance, Str_0) {
  Ref<Str> obj = Ref<Str>::New("Hello, World!");
  EXPECT_STREQ(obj->GetTypeKey(), "object.Str");
  EXPECT_TRUE(obj->IsInstance<Object>());
  EXPECT_FALSE(obj->IsInstance<Func>());
  EXPECT_TRUE(obj->IsInstance<Str>());
}

TEST(StaticTypeInheritance, Str_1) {
  Ref<Object> obj = Ref<Str>::New("Hello, World!");
  EXPECT_STREQ(obj->GetTypeKey(), "object.Str");
  EXPECT_TRUE(obj->IsInstance<Object>());
  EXPECT_FALSE(obj->IsInstance<Func>());
  EXPECT_TRUE(obj->IsInstance<Str>());
}

TEST(StaticTypeSubclass, NoException) {
  Ref<SubType> obj = Ref<SubType>::New(0);
  EXPECT_EQ(obj->data, 0);
}

TEST(StaticTypeSubclass, Exception) {
  try {
    Ref<SubType>::New(1);
    FAIL() << "No exception thrown";
  } catch (std::runtime_error& ex) {
    EXPECT_STREQ(ex.what(), "New Error");
  }
}

}  // namespace
