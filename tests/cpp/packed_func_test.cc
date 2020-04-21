/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/expr.h>

TEST(PackedFunc, Basic) {
  using namespace tvm;
  using namespace tvm::tir;
  using namespace tvm::runtime;
  int x = 0;
  void* handle = &x;
  DLTensor a;

  Var v = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.num_args == 3);
      CHECK(args.values[0].v_float64 == 1.0);
      CHECK(args.type_codes[0] == kDLFloat);
      CHECK(args.values[1].v_handle == &a);
      CHECK(args.type_codes[1] == kTVMDLTensorHandle);
      CHECK(args.values[2].v_handle == &x);
      CHECK(args.type_codes[2] == kTVMOpaqueHandle);
      *rv = Var("a");
    })(1.0, &a, handle);
  CHECK(v->name_hint == "a");
}

TEST(PackedFunc, Node) {
  using namespace tvm;
  using namespace tvm::tir;
  using namespace tvm::runtime;
  Var x;
  Var t = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.num_args == 1);
      CHECK(args[0].IsObjectRef<ObjectRef>());
      Var b = args[0];
      CHECK(x.same_as(b));
      *rv = b;
    })(x);
  CHECK(t.same_as(x));
}

TEST(PackedFunc, NDArray) {
  using namespace tvm;
  using namespace tvm::runtime;
  auto x = NDArray::Empty(
      {}, String2DLDataType("float32"),
      TVMContext{kDLCPU, 0});
  reinterpret_cast<float*>(x->data)[0] = 10.0f;
  CHECK(x.use_count() == 1);

  PackedFunc forward([&](TVMArgs args, TVMRetValue* rv) {
      *rv = args[0];
    });

  NDArray ret = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      NDArray y = args[0];
      DLTensor* ptr = args[0];
      CHECK(ptr == x.operator->());
      CHECK(x.same_as(y));
      CHECK(x.use_count() == 2);
      *rv = forward(y);
    })(x);
  CHECK(ret.use_count() == 2);
  CHECK(ret.same_as(x));
}

TEST(PackedFunc, str) {
  using namespace tvm;
  using namespace tvm::runtime;
  PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.num_args == 1);
      std::string x = args[0];
      CHECK(x == "hello");
      String y = args[0];
      CHECK(y == "hello");
      *rv = x;
    })("hello");

  PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK(args.num_args == 1);
      runtime::String s = args[0];
      CHECK(s == "hello");
  })(runtime::String("hello"));
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
      PrimExpr x = args[0];
      *rv = x.as<tvm::tir::IntImmNode>()->value + 1;
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
      DataType x = args[0];
      *rv = x;
    });
  auto get_type2 = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
      *rv = args[0];
    });
  CHECK(get_type("int32").operator DataType() == DataType::Int(32));
  CHECK(get_type("float").operator DataType() == DataType::Float(32));
  CHECK(get_type2("float32x2").operator DataType() == DataType::Float(32, 2));
}

TEST(TypedPackedFunc, HighOrder) {
  using namespace tvm;
  using namespace tvm::runtime;
  using Int1Func = TypedPackedFunc<int(int)>;
  using Int2Func = TypedPackedFunc<int(int, int)>;
  using BindFunc = TypedPackedFunc<Int1Func(Int2Func, int value)>;
  BindFunc ftyped;
  ftyped = [](Int2Func f1, int value) -> Int1Func {
    auto binded = [f1, value](int x) {
      return f1(value, x);
    };
    Int1Func x(binded);
    return x;
  };
  auto add = [](int x, int y) { return x + y; };
  CHECK_EQ(ftyped(Int2Func(add), 1)(2), 3);
  PackedFunc f = ftyped(Int2Func(add), 1);
  CHECK_EQ(f(3).operator int(), 4);
  // call the type erased version.
  Int1Func f1 = ftyped.packed()(Int2Func(add), 1);
  CHECK_EQ(f1(3), 4);
}

TEST(TypedPackedFunc, Deduce) {
  using namespace tvm::runtime;
  using tvm::runtime::detail::function_signature;

  TypedPackedFunc<int(float)> x;
  auto f = [](int x) -> int {
    return x + 1;
  };
  std::function<void(float)> y;

  static_assert(std::is_same<function_signature<decltype(x)>::FType,
                int(float)>::value, "invariant1");
  static_assert(std::is_same<function_signature<decltype(f)>::FType,
                int(int)>::value, "invariant2");
  static_assert(std::is_same<function_signature<decltype(y)>::FType,
                void(float)>::value, "invariant3");
}


TEST(PackedFunc, ObjectConversion) {
  using namespace tvm;
  using namespace tvm::tir;
  using namespace tvm::runtime;
  TVMRetValue rv;
  auto x = NDArray::Empty(
      {}, String2DLDataType("float32"),
      TVMContext{kDLCPU, 0});
  // assign null
  rv = ObjectRef();
  CHECK_EQ(rv.type_code(), kTVMNullptr);

  // Can assign NDArray to ret type
  rv = x;
  CHECK_EQ(rv.type_code(), kTVMNDArrayHandle);
  // Even if we assign base type it still shows as NDArray
  rv = ObjectRef(x);
  CHECK_EQ(rv.type_code(), kTVMNDArrayHandle);
  // Check convert back
  CHECK(rv.operator NDArray().same_as(x));
  CHECK(rv.operator ObjectRef().same_as(x));
  CHECK(!rv.IsObjectRef<PrimExpr>());

  auto pf1 = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args[0].type_code(), kTVMNDArrayHandle);
      CHECK(args[0].operator NDArray().same_as(x));
      CHECK(args[0].operator ObjectRef().same_as(x));
      CHECK(args[1].operator ObjectRef().get() == nullptr);
      CHECK(args[1].operator NDArray().get() == nullptr);
      CHECK(args[1].operator Module().get() == nullptr);
      CHECK(args[1].operator Array<NDArray>().get() == nullptr);
      CHECK(!args[0].IsObjectRef<PrimExpr>());
    });
  pf1(x, ObjectRef());
  pf1(ObjectRef(x), NDArray());

  // testcases for modules
  auto* pf = tvm::runtime::Registry::Get("runtime.SourceModuleCreate");
  CHECK(pf != nullptr);
  Module m = (*pf)("", "xyz");
  rv = m;
  CHECK_EQ(rv.type_code(), kTVMModuleHandle);
  // Even if we assign base type it still shows as NDArray
  rv = ObjectRef(m);
  CHECK_EQ(rv.type_code(), kTVMModuleHandle);
  // Check convert back
  CHECK(rv.operator Module().same_as(m));
  CHECK(rv.operator ObjectRef().same_as(m));
  CHECK(!rv.IsObjectRef<NDArray>());

  auto pf2 = PackedFunc([&](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args[0].type_code(), kTVMModuleHandle);
      CHECK(args[0].operator Module().same_as(m));
      CHECK(args[0].operator ObjectRef().same_as(m));
      CHECK(args[1].operator ObjectRef().get() == nullptr);
      CHECK(args[1].operator NDArray().get() == nullptr);
      CHECK(args[1].operator Module().get() == nullptr);
      CHECK(!args[0].IsObjectRef<PrimExpr>());
    });
  pf2(m, ObjectRef());
  pf2(ObjectRef(m), Module());
}

TEST(TypedPackedFunc, RValue) {
  using namespace tvm;
  using namespace tvm::runtime;
  {

    auto inspect = [](TVMArgs args, TVMRetValue* rv) {
      for (int i = 0; i < args.size(); ++i) {
        CHECK_EQ(args[0].type_code(), kTVMObjectRValueRefArg);
      }
    };
    PackedFunc  finspect(inspect);
    finspect(tir::Var("x"));
  }
  {
    auto f = [](tir::Var x, bool move) {
      if (move) {
        CHECK(x.unique());
      } else {
        CHECK(!x.unique());
      }
      CHECK(x->name_hint == "x");
      return x;
    };
    TypedPackedFunc<tir::Var(tir::Var, bool)> tf(f);

    tir::Var var("x");
    CHECK(var.unique());
    tf(var, false);
    // move the result to the function.
    tir::Var ret = tf(std::move(var), true);
    CHECK(!var.defined());
  }

  {
    // pass child class.
    auto f = [](PrimExpr x, bool move) {
      if (move) {
        CHECK(x.unique());
      } else {
        CHECK(!x.unique());
      }
      return x;
    };
    TypedPackedFunc<PrimExpr(PrimExpr, bool)> tf(f);

    tir::Var var("x");
    CHECK(var.unique());
    tf(var, false);
    tf(std::move(var), true);
    // auto conversion.
    tf(1, true);
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
