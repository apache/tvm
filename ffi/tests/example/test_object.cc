#include <gtest/gtest.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>

namespace {

using namespace tvm::ffi;

class IntObj : public Object {
 public:
  int64_t value;

  IntObj(int64_t value) : value(value) {}
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

}  // namespace
