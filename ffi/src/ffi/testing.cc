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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/reflection.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace tvm {
namespace ffi {

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

void TestRaiseError(String kind, String msg) {
  throw ffi::Error(kind, msg, TVM_FFI_TRACEBACK_HERE);
}

void TestApply(Function f, PackedArgs args, Any* ret) { f.CallPacked(args, ret); }

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
      .def_packed("testing.apply",
                  [](PackedArgs args, Any* ret) {
                    auto f = args[0].cast<Function>();
                    TestApply(f, args.Slice(1), ret);
                  })
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
