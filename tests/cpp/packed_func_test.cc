#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tvm.h>
#include <tvm/ir.h>

TEST(PackedFunc, Basic) {
  using namespace tvm;
  using namespace tvm::runtime;
  int x = 0;
  void* handle = &x;
  TVMArray a;

  Var v = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.num_args == 3);
      CHECK(args.values[0].v_float64 == 1.0);
      CHECK(args.type_codes[0] == kDLFloat);
      CHECK(args.values[1].v_handle == &a);
      CHECK(args.type_codes[1] == kArrayHandle);
      CHECK(args.values[2].v_handle == &x);
      CHECK(args.type_codes[2] == kHandle);
      *rv = Var("a");
    })(1.0, &a, handle);
  CHECK(v->name_hint == "a");
}

TEST(PackedFunc, Node) {
  using namespace tvm;
  using namespace tvm::runtime;
  Var x;
  Var t = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.num_args == 1);
      CHECK(args.type_codes[0] == kNodeHandle);
      Var b = args[0];
      CHECK(x.same_as(b));
      *rv = b;
    })(x);
  CHECK(t.same_as(x));
}

TEST(PackedFunc, str) {
  using namespace tvm;
  using namespace tvm::runtime;
  PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.num_args == 1);
      std::string x = args[0];
      CHECK(x == "hello");
      *rv = x;
    })("hello");
}


TEST(PackedFunc, func) {
  using namespace tvm;
  using namespace tvm::runtime;
  PackedFunc addone([&](TVMArgs args, TVMRetValue* rv) {
      *rv = args[0].operator int() + 1;
    });
  // function as arguments
  int r0 = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      PackedFunc f = args[0];
      // TVMArgValue -> Arguments as function
      *rv = f(args[1]).operator int();
    })(addone, 1);
  CHECK_EQ(r0, 2);

  int r1 = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      // TVMArgValue -> TVMRetValue
      *rv = args[1];
    })(2, 100);
  CHECK_EQ(r1, 100);

  int r2 = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      // re-assignment
      *rv = args[0];
      // TVMRetValue -> Function argument
      *rv = addone(args[0].operator PackedFunc()(args[1], 1));
    })(addone, 100);
  CHECK_EQ(r2, 102);
}

TEST(PackedFunc, Expr) {
  using namespace tvm;
  using namespace tvm::runtime;
  // automatic conversion of int to expr
  PackedFunc addone([](TVMArgs args, TVMRetValue* rv) {
      Expr x = args[0];
      *rv = x.as<tvm::ir::IntImm>()->value + 1;
  });
  int r0 = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      PackedFunc f = args[0];
      // TVMArgValue -> Arguments as function
      *rv = f(args[1]).operator int();
    })(addone, 1);
  CHECK_EQ(r0, 2);
}

TEST(PackedFunc, Type) {
  using namespace tvm;
  using namespace tvm::runtime;
  auto get_type = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      Type x = args[0];
      *rv = x;
    });
  auto get_type2 = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      *rv = args[0];
    });
  CHECK(get_type("int32").operator Type() == Int(32));
  CHECK(get_type("float").operator Type() == Float(32));
  CHECK(get_type2("float32x2").operator Type() == Float(32, 2));
}

// new namespoace
namespace test {
// register int vector as extension type
using IntVector = std::vector<int>;
}  // namespace test

namespace tvm {
namespace runtime {

template<>
struct extension_class_info<test::IntVector> {
  static const int code = kExtBegin + 1;
};
}  // runtime
}  // tvm

// do registration, this need to be in cc file
TVM_REGISTER_EXT_TYPE(test::IntVector);

TEST(PackedFunc, ExtensionType) {
  using namespace tvm;
  using namespace tvm::runtime;
  // note: class are copy by value.
  test::IntVector vec{1, 2, 4};

  auto copy_vec = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      // copy by value
      const test::IntVector& v = args[0].AsExtension<test::IntVector>();
      CHECK(&v == &vec);
      test::IntVector v2 = args[0];
      CHECK_EQ(v2.size(), 3U);
      CHECK_EQ(v[2], 4);
      // return copy by value
      *rv = v2;
    });

  auto pass_vec = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      // copy by value
      *rv = args[0];
    });

  test::IntVector vret1 = copy_vec(vec);
  test::IntVector vret2 = pass_vec(copy_vec(vec));
  CHECK_EQ(vret1.size(), 3U);
  CHECK_EQ(vret2.size(), 3U);
  CHECK_EQ(vret1[2], 4);
  CHECK_EQ(vret2[2], 4);
}


int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
