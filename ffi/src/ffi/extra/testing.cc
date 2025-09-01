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
// This file is used for testing the FFI API.
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace tvm {
namespace ffi {

// Step 1: Define the object class (stores the actual data)
class TestIntPairObj : public tvm::ffi::Object {
 public:
  int64_t a;
  int64_t b;

  TestIntPairObj() = default;
  TestIntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  // Required: declare type information
  static constexpr const char* _type_key = "testing.TestIntPair";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(TestIntPairObj, tvm::ffi::Object);
};

// Step 2: Define the reference wrapper (user-facing interface)
class TestIntPair : public tvm::ffi::ObjectRef {
 public:
  // Constructor
  explicit TestIntPair(int64_t a, int64_t b) {
    data_ = tvm::ffi::make_object<TestIntPairObj>(a, b);
  }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS(TestIntPair, tvm::ffi::ObjectRef, TestIntPairObj);
};

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestIntPairObj>()
      .def_ro("a", &TestIntPairObj::a)
      .def_ro("b", &TestIntPairObj::b)
      .def_static("__create__",
                  [](int64_t a, int64_t b) -> TestIntPair { return TestIntPair(a, b); });
});

class TestObjectBase : public Object {
 public:
  int64_t v_i64;
  double v_f64;
  String v_str;

  int64_t AddI64(int64_t other) const { return v_i64 + other; }

  // declare as one slot, with float as overflow
  static constexpr bool _type_mutable = true;
  static constexpr uint32_t _type_child_slots = 1;
  static constexpr const char* _type_key = "testing.TestObjectBase";
  TVM_FFI_DECLARE_BASE_OBJECT_INFO(TestObjectBase, Object);
};

class TestObjectDerived : public TestObjectBase {
 public:
  Map<Any, Any> v_map;
  Array<Any> v_array;

  // declare as one slot, with float as overflow
  static constexpr const char* _type_key = "testing.TestObjectDerived";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(TestObjectDerived, TestObjectBase);
};

TVM_FFI_NO_INLINE void TestRaiseError(String kind, String msg) {
  // keep name and no liner for testing traceback
  throw ffi::Error(kind, msg, TVMFFITraceback(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0));
}

TVM_FFI_NO_INLINE void TestApply(PackedArgs args, Any* ret) {
  // keep name and no liner for testing traceback
  auto f = args[0].cast<Function>();
  f.CallPacked(args.Slice(1), ret);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<TestObjectBase>()
      .def_rw("v_i64", &TestObjectBase::v_i64, refl::DefaultValue(10), "i64 field")
      .def_ro("v_f64", &TestObjectBase::v_f64, refl::DefaultValue(10.0))
      .def_rw("v_str", &TestObjectBase::v_str, refl::DefaultValue("hello"))
      .def("add_i64", &TestObjectBase::AddI64, "add_i64 method");

  refl::ObjectDef<TestObjectDerived>()
      .def_ro("v_map", &TestObjectDerived::v_map)
      .def_ro("v_array", &TestObjectDerived::v_array);

  refl::GlobalDef()
      .def("testing.test_raise_error", TestRaiseError)
      .def_packed("testing.nop", [](PackedArgs args, Any* ret) { *ret = args[0]; })
      .def_packed("testing.echo", [](PackedArgs args, Any* ret) { *ret = args[0]; })
      .def_packed("testing.apply", TestApply)
      .def("testing.run_check_signal",
           [](int nsec) {
             for (int i = 0; i < nsec; ++i) {
               if (TVMFFIEnvCheckSignals() != 0) {
                 throw ffi::EnvErrorAlreadySet();
               }
               std::this_thread::sleep_for(std::chrono::seconds(1));
             }
             std::cout << "Function finished without catching signal" << std::endl;
           })
      .def("testing.object_use_count", [](const Object* obj) { return obj->use_count(); });
});

}  // namespace ffi
}  // namespace tvm
