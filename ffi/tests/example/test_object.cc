#include <gtest/gtest.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>

namespace {

using namespace tvm::ffi;

class NumberObj : public Object {
 public:
  // declare as one slot, with float as overflow
  static constexpr uint32_t _type_child_slots = 1;
  static constexpr const char* _type_key = "test.Number";
  TVM_FFI_DECLARE_BASE_OBJECT_INFO(NumberObj, Object);
};

class Number : public ObjectRef {
 public:
  TVM_FFI_DEFINE_NULLABLE_OBJECT_REF_METHODS(Number, ObjectRef, NumberObj);
};

class IntObj : public NumberObj {
 public:
  int64_t value;

  IntObj(int64_t value) : value(value) {}

  static constexpr const char* _type_key = "test.Int";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(IntObj, NumberObj);
};

class Int : public Number {
 public:
  explicit Int(int64_t value) {
    data_ = make_object<IntObj>(value);
  }

  TVM_FFI_DEFINE_NULLABLE_OBJECT_REF_METHODS(Int, Number, IntObj);
};

class FloatObj : public NumberObj {
 public:
  double value;

  FloatObj(double value) : value(value) {}

  static constexpr const char* _type_key = "test.Float";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(FloatObj, NumberObj);
};

class Float : public Number {
 public:
  explicit Float(double value) {
    data_ = make_object<FloatObj>(value);
  }

  TVM_FFI_DEFINE_NULLABLE_OBJECT_REF_METHODS(Float, Number, FloatObj);
};

TEST(Object, RefCounter) {
  ObjectPtr<IntObj> a = make_object<IntObj>(11);
  ObjectPtr<IntObj> b = a;

  EXPECT_EQ(a->value, 11);

  EXPECT_EQ(a.use_count(), 2);
  b.reset();
  EXPECT_EQ(a.use_count(), 1);
  EXPECT_TRUE(b == nullptr);
  EXPECT_EQ(b.use_count(), 0);

  ObjectPtr<IntObj> c = std::move(a);
  EXPECT_EQ(c.use_count(), 1);
  EXPECT_TRUE(a == nullptr);

  EXPECT_EQ(c->value, 11);
}

TEST(Object, TypeInfo) {
  const TypeInfo* info = tvm::ffi::details::ObjectGetTypeInfo(IntObj::RuntimeTypeIndex());
  EXPECT_TRUE(info != nullptr);
  EXPECT_EQ(info->type_index, IntObj::RuntimeTypeIndex());
  EXPECT_EQ(info->type_depth, 2);
  EXPECT_EQ(info->type_acenstors[0], Object::_type_index);
  EXPECT_EQ(info->type_acenstors[1], NumberObj::_type_index);
  EXPECT_GE(info->type_index, TypeIndex::kTVMFFIDynObjectBegin);
}

TEST(Object, InstanceCheck) {
  ObjectPtr<Object> a = make_object<IntObj>(11);
  ObjectPtr<Object> b = make_object<FloatObj>(11);

  EXPECT_TRUE(a->IsInstance<Object>());
  EXPECT_TRUE(a->IsInstance<NumberObj>());
  EXPECT_TRUE(a->IsInstance<IntObj>());
  EXPECT_TRUE(!a->IsInstance<FloatObj>());

  EXPECT_TRUE(a->IsInstance<Object>());
  EXPECT_TRUE(b->IsInstance<NumberObj>());
  EXPECT_TRUE(!b->IsInstance<IntObj>());
  EXPECT_TRUE(b->IsInstance<FloatObj>());
}

TEST(ObjectRef, as) {
  ObjectRef a = Int(10);
  ObjectRef b = Float(20);
  // nullable object
  ObjectRef c(nullptr);

  EXPECT_TRUE(a.as<IntObj>() != nullptr);
  EXPECT_TRUE(a.as<FloatObj>() == nullptr);
  EXPECT_TRUE(a.as<NumberObj>() != nullptr);

  EXPECT_TRUE(b.as<IntObj>() == nullptr);
  EXPECT_TRUE(b.as<FloatObj>() != nullptr);
  EXPECT_TRUE(b.as<NumberObj>() != nullptr);

  EXPECT_TRUE(c.as<IntObj>() == nullptr);
  EXPECT_TRUE(c.as<FloatObj>() == nullptr);
  EXPECT_TRUE(c.as<NumberObj>() == nullptr);

  EXPECT_EQ(a.as<IntObj>()->value, 10);
  EXPECT_EQ(b.as<FloatObj>()->value, 20);
}

}  // namespace
