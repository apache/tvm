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

/*!
 *  FFI registration code used for frontend testing purposes.
 * \file ffi_testing.cc
 */
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/runtime/module.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <chrono>
#include <thread>

namespace tvm {
// Attrs used to python API
struct TestAttrs : public AttrsNodeReflAdapter<TestAttrs> {
  int axis;
  ffi::String name;
  ffi::Array<PrimExpr> padding;
  TypedEnvFunc<int(int)> func;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TestAttrs>()
        .def_ro("axis", &TestAttrs::axis, "axis field", refl::DefaultValue(10))
        .def_ro("name", &TestAttrs::name, "name")
        .def_ro("padding", &TestAttrs::padding, "padding of input",
                refl::DefaultValue(ffi::Array<PrimExpr>({0, 0})))
        .def_ro("func", &TestAttrs::func, "some random env function",
                refl::DefaultValue(TypedEnvFunc<int(int)>(nullptr)));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("attrs.TestAttrs", TestAttrs, BaseAttrsNode);
};

TVM_FFI_STATIC_INIT_BLOCK() { TestAttrs::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("testing.GetShapeSize",
           [](ffi::Shape shape) { return static_cast<int64_t>(shape.size()); })
      .def("testing.GetShapeElem",
           [](ffi::Shape shape, int idx) {
             ICHECK_LT(idx, shape.size());
             return shape[idx];
           })
      .def_packed("testing.test_wrap_callback",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    ffi::Function pf = args[0].cast<ffi::Function>();
                    *ret = ffi::TypedFunction<void()>([pf]() { pf(); });
                  })
      .def_packed("testing.test_wrap_callback_suppress_err",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    ffi::Function pf = args[0].cast<ffi::Function>();
                    auto result = ffi::TypedFunction<void()>([pf]() {
                      try {
                        pf();
                      } catch (std::exception& err) {
                      }
                    });
                    *ret = result;
                  })
      .def_packed("testing.test_check_eq_callback",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    auto msg = args[0].cast<std::string>();
                    *ret = ffi::TypedFunction<void(int x, int y)>(
                        [msg](int x, int y) { CHECK_EQ(x, y) << msg; });
                  })
      .def_packed("testing.device_test",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    auto dev = args[0].cast<Device>();
                    int dtype = args[1].cast<int>();
                    int did = args[2].cast<int>();
                    CHECK_EQ(static_cast<int>(dev.device_type), dtype);
                    CHECK_EQ(static_cast<int>(dev.device_id), did);
                    *ret = dev;
                  })
      .def_packed("testing.identity_cpp", [](ffi::PackedArgs args, ffi::Any* ret) {
        const auto identity_func = tvm::ffi::Function::GetGlobal("testing.identity_py");
        ICHECK(identity_func.has_value())
            << "AttributeError: \"testing.identity_py\" is not registered. Please check "
               "if the python module is properly loaded";
        *ret = (*identity_func)(args[0]);
      });
}

// in src/api_test.cc
void ErrorTest(int x, int y) {
  // raise ValueError
  CHECK_EQ(x, y) << "ValueError: expect x and y to be equal.";
  if (x == 1) {
    // raise InternalError.
    LOG(FATAL) << "InternalError: cannot reach here";
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("testing.ErrorTest", ErrorTest);
}

class FrontendTestModuleNode : public ffi::ModuleObj {
 public:
  const char* kind() const final { return "frontend_test"; }

  static constexpr const char* kAddFunctionName = "__add_function";

  virtual ffi::Optional<ffi::Function> GetFunction(const ffi::String& name);

 private:
  std::unordered_map<std::string, ffi::Function> functions_;
};

constexpr const char* FrontendTestModuleNode::kAddFunctionName;

ffi::Optional<ffi::Function> FrontendTestModuleNode::GetFunction(const ffi::String& name) {
  ffi::Module self_strong_ref = ffi::GetRef<ffi::Module>(this);
  if (name == kAddFunctionName) {
    return ffi::Function::FromTyped(
        [this, self_strong_ref](std::string func_name, ffi::Function pf) {
          CHECK_NE(func_name, kAddFunctionName)
              << "func_name: cannot be special function " << kAddFunctionName;
          functions_[func_name] = pf;
        });
  }

  auto it = functions_.find(name);
  if (it == functions_.end()) {
    return std::nullopt;
  }

  return it->second;
}

ffi::Module NewFrontendTestModule() {
  auto n = ffi::make_object<FrontendTestModuleNode>();
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("testing.FrontendTestModule", NewFrontendTestModule)
      .def(
          "testing.sleep_in_ffi",
          [](double timeout) {
            std::chrono::duration<int64_t, std::nano> duration(static_cast<int64_t>(timeout * 1e9));
            std::this_thread::sleep_for(duration);
          })
      .def("testing.ReturnsVariant",
           [](int x) -> ffi::Variant<ffi::String, IntImm> {
             if (x % 2 == 0) {
               return IntImm(DataType::Int(64), x / 2);
             } else {
               return ffi::String("argument was odd");
             }
           })
      .def("testing.AcceptsVariant",
           [](ffi::Variant<ffi::String, Integer> arg) -> ffi::String {
             if (auto opt_str = arg.as<ffi::String>()) {
               return ffi::StaticTypeKey::kTVMFFIStr;
             } else {
               return arg.get<Integer>().GetTypeKey();
             }
           })
      .def("testing.AcceptsBool", [](bool arg) -> bool { return arg; })
      .def("testing.AcceptsInt", [](int arg) -> int { return arg; })
      .def("testing.AcceptsObjectRefArray", [](ffi::Array<Any> arg) -> Any { return arg[0]; })
      .def("testing.AcceptsMapReturnsValue",
           [](ffi::Map<Any, Any> map, Any key) -> Any { return map[key]; })
      .def("testing.AcceptsMapReturnsMap", [](ffi::Map<Any, Any> map) -> ObjectRef { return map; })
      .def("testing.AcceptsPrimExpr", [](PrimExpr expr) -> ObjectRef { return expr; })
      .def("testing.AcceptsArrayOfPrimExpr",
           [](ffi::Array<PrimExpr> arr) -> ObjectRef {
             for (ObjectRef item : arr) {
               CHECK(item->IsInstance<PrimExprNode>()) << "Array contained " << item->GetTypeKey()
                                                       << " when it should contain PrimExpr";
             }
             return arr;
           })
      .def("testing.AcceptsArrayOfVariant",
           [](ffi::Array<ffi::Variant<ffi::Function, PrimExpr>> arr) -> ObjectRef {
             for (auto item : arr) {
               CHECK(item.as<PrimExpr>() || item.as<ffi::Function>())
                   << "Array should contain either PrimExpr or ffi::Function";
             }
             return arr;
           })
      .def("testing.AcceptsMapOfPrimExpr", [](ffi::Map<Any, PrimExpr> map) -> ObjectRef {
        for (const auto& kv : map) {
          ObjectRef value = kv.second;
          CHECK(value->IsInstance<PrimExprNode>())
              << "Map contained " << value->GetTypeKey() << " when it should contain PrimExpr";
        }
        return map;
      });
}

/**
 * Simple event logger that can be used for testing purposes
 */
class TestingEventLogger {
 public:
  struct Entry {
    ffi::String event;
    double time_us;
  };

  TestingEventLogger() {
    entries_.reserve(1024);
    start_ = std::chrono::high_resolution_clock::now();
  }

  void Record(ffi::String event) {
    auto tend = std::chrono::high_resolution_clock::now();
    double time_us = static_cast<double>((tend - start_).count()) / 1e3;
    entries_.emplace_back(Entry{event, time_us});
  }

  void Reset() { entries_.clear(); }

  void Dump() const {
    for (const Entry& e : entries_) {
      LOG(INFO) << e.event << "\t" << e.time_us << " us";
    }
  }

  static TestingEventLogger* ThreadLocal() {
    thread_local TestingEventLogger inst;
    return &inst;
  }

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::vector<Entry> entries_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("testing.record_event",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    if (args.size() != 0 && args[0].try_cast<ffi::String>()) {
                      TestingEventLogger::ThreadLocal()->Record(args[0].cast<ffi::String>());
                    } else {
                      TestingEventLogger::ThreadLocal()->Record("X");
                    }
                  })
      .def_packed(
          "testing.reset_events",
          [](ffi::PackedArgs args, ffi::Any* rv) { TestingEventLogger::ThreadLocal()->Reset(); })
      .def("testing.dump_events", []() { TestingEventLogger::ThreadLocal()->Dump(); });
}
}  // namespace tvm
