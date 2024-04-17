#include <gtest/gtest.h>
#include <tvm/ffi/ffi.hpp>

#if TVM_FFI_ALLOW_DYN_TYPE == 1

namespace {
using namespace tvm::ffi;

struct TestObj : public Object {
  int x;
  explicit TestObj(int x) : x(x) {}
  TVM_FFI_DEF_DYN_TYPE(TestObj, Object, "test.TestObj");
};

struct SubTestObj : public TestObj {
  int y;
  explicit SubTestObj(int x, int y) : TestObj(x), y(y) {}
  TVM_FFI_DEF_DYN_TYPE(SubTestObj, TestObj, "test.SubTestObj");
};

void CheckAncestor(int32_t num, const int32_t *ancestors,
                   std::vector<int32_t> expected) {
  EXPECT_EQ(num, expected.size());
  for (int i = 0; i < num; ++i) {
    EXPECT_EQ(ancestors[i], expected[i]);
  }
}

TEST(DynTypeInfo, TestObj) {
  EXPECT_GE(TestObj::_type_index,
            static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDynObjectBegin));
  EXPECT_STRCASEEQ(TestObj::_type_key, "test.TestObj");
  EXPECT_EQ(TestObj::_type_depth, 1);
  CheckAncestor(TestObj::_type_depth, TestObj::_type_ancestors.data(),
                {static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject)});
}

TEST(DynTypeInfo, SubTestObj) {
  EXPECT_GE(SubTestObj::_type_index,
            static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDynObjectBegin));
  EXPECT_NE(SubTestObj::_type_index, TestObj::_type_index);
  EXPECT_STRCASEEQ(SubTestObj::_type_key, "test.SubTestObj");
  EXPECT_EQ(SubTestObj::_type_depth, 2);
  CheckAncestor(SubTestObj::_type_depth, SubTestObj::_type_ancestors.data(),
                {static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIObject),
                 TestObj::_type_index});
}

TEST(DynTypeInheritance, TestObj) {
  Ref<TestObj> obj = Ref<TestObj>::New(10);
  EXPECT_EQ(obj->x, 10);
  EXPECT_TRUE(obj->IsInstance<Object>());
  EXPECT_TRUE(obj->IsInstance<TestObj>());
  EXPECT_FALSE(obj->IsInstance<Func>());
  EXPECT_FALSE(obj->IsInstance<Str>());
}

TEST(DynTypeInheritance, SubTestObj) {
  Ref<SubTestObj> obj = Ref<SubTestObj>::New(10, 20);
  EXPECT_EQ(obj->x, 10);
  EXPECT_EQ(obj->y, 20);
  EXPECT_TRUE(obj->IsInstance<Object>());
  EXPECT_TRUE(obj->IsInstance<TestObj>());
  EXPECT_TRUE(obj->IsInstance<SubTestObj>());
  EXPECT_FALSE(obj->IsInstance<Func>());
  EXPECT_FALSE(obj->IsInstance<Str>());
}

} // namespace

#endif
