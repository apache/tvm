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
#include <gtest/gtest.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

// test-cases used in example code
namespace {

void ExampleAny() {
  namespace ffi = tvm::ffi;
  // Create an Any from various types
  ffi::Any int_value = 42;
  ffi::Any float_value = 3.14;
  ffi::Any string_value = "hello world";

  // AnyView provides a lightweight view without ownership
  ffi::AnyView view = int_value;
  // we can cast Any/AnyView to a specific type
  int extracted = view.cast<int>();
  EXPECT_EQ(extracted, 42);

  // If we are not sure about the type
  // we can use as to get an optional value
  std::optional<int> maybe_int = view.as<int>();
  if (maybe_int.has_value()) {
    EXPECT_EQ(maybe_int.value(), 42);
  }
  // Try cast is another version that will try to run the type
  // conversion even if the type does not exactly match
  std::optional<int> maybe_int_try = view.try_cast<int>();
  if (maybe_int_try.has_value()) {
    EXPECT_EQ(maybe_int_try.value(), 42);
  }
}

TEST(Example, Any) { ExampleAny(); }

void ExampleFunctionFromPacked() {
  namespace ffi = tvm::ffi;
  // Create a function from a typed lambda
  ffi::Function fadd1 =
      ffi::Function::FromPacked([](const ffi::AnyView* args, int32_t num_args, ffi::Any* rv) {
        TVM_FFI_ICHECK_EQ(num_args, 1);
        int a = args[0].cast<int>();
        *rv = a + 1;
      });
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);
}

void ExampleFunctionFromTyped() {
  namespace ffi = tvm::ffi;
  // Create a function from a typed lambda
  ffi::Function fadd1 = ffi::Function::FromTyped([](const int a) -> int { return a + 1; });
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);
}

void ExampleFunctionPassFunction() {
  namespace ffi = tvm::ffi;
  // Create a function from a typed lambda
  ffi::Function fapply = ffi::Function::FromTyped(
      [](const ffi::Function f, ffi::Any param) { return f(param.cast<int>()); });
  ffi::Function fadd1 = ffi::Function::FromTyped(  //
      [](const int a) -> int { return a + 1; });
  int b = fapply(fadd1, 2).cast<int>();
  EXPECT_EQ(b, 3);
}

void ExamplegGlobalFunctionRegistry() {
  namespace ffi = tvm::ffi;
  ffi::reflection::GlobalDef().def("xyz.add1", [](const int a) -> int { return a + 1; });
  ffi::Function fadd1 = ffi::Function::GetGlobalRequired("xyz.add1");
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);
}

void FuncThrowError() {
  namespace ffi = tvm::ffi;
  TVM_FFI_THROW(TypeError) << "test0";
}

void ExampleErrorHandling() {
  namespace ffi = tvm::ffi;
  try {
    FuncThrowError();
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "TypeError");
    EXPECT_EQ(e.message(), "test0");
    std::cout << e.traceback() << std::endl;
  }
}

TEST(Example, Function) {
  ExampleFunctionFromPacked();
  ExampleFunctionFromTyped();
  ExampleFunctionPassFunction();
  ExamplegGlobalFunctionRegistry();
  ExampleErrorHandling();
}

struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) { tensor->data = malloc(tvm::ffi::GetDataSize(*tensor)); }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

void ExampleTensor() {
  namespace ffi = tvm::ffi;
  ffi::Shape shape = {1, 2, 3};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};
  ffi::Tensor tensor = ffi::Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
}

void ExampleTensorDLPack() {
  namespace ffi = tvm::ffi;
  ffi::Shape shape = {1, 2, 3};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};
  ffi::Tensor tensor = ffi::Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
  // convert to DLManagedTensorVersioned
  DLManagedTensorVersioned* dlpack = tensor.ToDLPackVersioned();
  // load back from DLManagedTensorVersioned
  ffi::Tensor tensor2 = ffi::Tensor::FromDLPackVersioned(dlpack);
}

TEST(Example, Tensor) {
  ExampleTensor();
  ExampleTensorDLPack();
}

void ExampleString() {
  namespace ffi = tvm::ffi;
  ffi::String str = "hello world";
  EXPECT_EQ(str.size(), 11);
  std::string std_str = str;
  EXPECT_EQ(std_str, "hello world");
}

TEST(Example, String) { ExampleString(); }

void ExampleArray() {
  namespace ffi = tvm::ffi;
  ffi::Array<int> numbers = {1, 2, 3};
  EXPECT_EQ(numbers.size(), 3);
  EXPECT_EQ(numbers[0], 1);

  ffi::Function head = ffi::Function::FromTyped([](const ffi::Array<int> a) { return a[0]; });
  EXPECT_EQ(head(numbers).cast<int>(), 1);

  try {
    // throw an error because 2.2 is not int
    head(ffi::Array<ffi::Any>({1, 2.2}));
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "TypeError");
  }
}

void ExampleTuple() {
  namespace ffi = tvm::ffi;
  ffi::Tuple<int, ffi::String, bool> tup(42, "hello", true);

  EXPECT_EQ(tup.get<0>(), 42);
  EXPECT_EQ(tup.get<1>(), "hello");
  EXPECT_EQ(tup.get<2>(), true);
}

TEST(Example, Array) {
  ExampleArray();
  ExampleTuple();
}

void ExampleMap() {
  namespace ffi = tvm::ffi;

  ffi::Map<ffi::String, int> map0 = {{"Alice", 100}, {"Bob", 95}};

  EXPECT_EQ(map0.size(), 2);
  EXPECT_EQ(map0.at("Alice"), 100);
  EXPECT_EQ(map0.count("Alice"), 1);
}

TEST(Example, Map) { ExampleMap(); }

void ExampleOptional() {
  namespace ffi = tvm::ffi;
  ffi::Optional<int> opt0 = 100;
  EXPECT_EQ(opt0.has_value(), true);
  EXPECT_EQ(opt0.value(), 100);

  ffi::Optional<ffi::String> opt1;
  EXPECT_EQ(opt1.has_value(), false);
  EXPECT_EQ(opt1.value_or("default"), "default");
}

TEST(Example, Optional) { ExampleOptional(); }

void ExampleVariant() {
  namespace ffi = tvm::ffi;
  ffi::Variant<int, ffi::String> var0 = 100;
  EXPECT_EQ(var0.get<int>(), 100);

  var0 = ffi::String("hello");
  std::optional<ffi::String> maybe_str = var0.as<ffi::String>();
  EXPECT_EQ(maybe_str.value(), "hello");

  std::optional<int> maybe_int2 = var0.as<int>();
  EXPECT_EQ(maybe_int2.has_value(), false);
}

TEST(Example, Variant) { ExampleVariant(); }

// Step 1: Define the object class (stores the actual data)
class MyIntPairObj : public tvm::ffi::Object {
 public:
  int64_t a;
  int64_t b;

  MyIntPairObj() = default;
  MyIntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  // Required: declare type information
  static constexpr const char* _type_key = "example.MyIntPair";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(MyIntPairObj, tvm::ffi::Object);
};

// Step 2: Define the reference wrapper (user-facing interface)
class MyIntPair : public tvm::ffi::ObjectRef {
 public:
  // Constructor
  explicit MyIntPair(int64_t a, int64_t b) { data_ = tvm::ffi::make_object<MyIntPairObj>(a, b); }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS(MyIntPair, tvm::ffi::ObjectRef, MyIntPairObj);
};

void ExampleObjectPtr() {
  namespace ffi = tvm::ffi;
  ffi::ObjectPtr<MyIntPairObj> obj = ffi::make_object<MyIntPairObj>(100, 200);
  EXPECT_EQ(obj->a, 100);
  EXPECT_EQ(obj->b, 200);
}

void ExampleObjectRef() {
  namespace ffi = tvm::ffi;
  MyIntPair pair(100, 200);
  EXPECT_EQ(pair->a, 100);
  EXPECT_EQ(pair->b, 200);
}

void ExampleObjectRefAny() {
  namespace ffi = tvm::ffi;
  MyIntPair pair(100, 200);
  ffi::Any any = pair;
  MyIntPair pair2 = any.cast<MyIntPair>();
  EXPECT_EQ(pair2->a, 100);
  EXPECT_EQ(pair2->b, 200);
}

TEST(Example, ObjectPtr) {
  ExampleObjectPtr();
  ExampleObjectRef();
  ExampleObjectRefAny();
}

}  // namespace
