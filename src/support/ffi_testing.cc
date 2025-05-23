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
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/runtime/module.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

#include <chrono>
#include <thread>

namespace tvm {
// Attrs used to python API
struct TestAttrs : public AttrsNode<TestAttrs> {
  int axis;
  String name;
  Array<PrimExpr> padding;
  TypedEnvFunc<int(int)> func;

  TVM_DECLARE_ATTRS(TestAttrs, "attrs.TestAttrs") {
    TVM_ATTR_FIELD(axis).set_default(10).set_lower_bound(1).set_upper_bound(10).describe(
        "axis field");
    TVM_ATTR_FIELD(name).describe("name");
    TVM_ATTR_FIELD(padding).describe("padding of input").set_default(Array<PrimExpr>({0, 0}));
    TVM_ATTR_FIELD(func)
        .describe("some random env function")
        .set_default(TypedEnvFunc<int(int)>(nullptr));
  }
};

TVM_REGISTER_NODE_TYPE(TestAttrs);

TVM_FFI_REGISTER_GLOBAL("testing.GetShapeSize").set_body_typed([](ffi::Shape shape) {
  return static_cast<int64_t>(shape.size());
});

TVM_FFI_REGISTER_GLOBAL("testing.GetShapeElem").set_body_typed([](ffi::Shape shape, int idx) {
  ICHECK_LT(idx, shape.size());
  return shape[idx];
});

TVM_FFI_REGISTER_GLOBAL("testing.test_wrap_callback")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      ffi::Function pf = args[0].cast<ffi::Function>();
      *ret = ffi::TypedFunction<void()>([pf]() { pf(); });
    });

TVM_FFI_REGISTER_GLOBAL("testing.test_wrap_callback_suppress_err")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      ffi::Function pf = args[0].cast<ffi::Function>();
      auto result = ffi::TypedFunction<void()>([pf]() {
        try {
          pf();
        } catch (std::exception& err) {
        }
      });
      *ret = result;
    });

TVM_FFI_REGISTER_GLOBAL("testing.test_check_eq_callback")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      auto msg = args[0].cast<std::string>();
      *ret = ffi::TypedFunction<void(int x, int y)>([msg](int x, int y) { CHECK_EQ(x, y) << msg; });
    });

TVM_FFI_REGISTER_GLOBAL("testing.device_test")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      auto dev = args[0].cast<Device>();
      int dtype = args[1].cast<int>();
      int did = args[2].cast<int>();
      CHECK_EQ(static_cast<int>(dev.device_type), dtype);
      CHECK_EQ(static_cast<int>(dev.device_id), did);
      *ret = dev;
    });

TVM_FFI_REGISTER_GLOBAL("testing.identity_cpp")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      const auto identity_func = tvm::ffi::Function::GetGlobal("testing.identity_py");
      ICHECK(identity_func.has_value())
          << "AttributeError: \"testing.identity_py\" is not registered. Please check "
             "if the python module is properly loaded";
      *ret = (*identity_func)(args[0]);
    });

// in src/api_test.cc
void ErrorTest(int x, int y) {
  // raise ValueError
  CHECK_EQ(x, y) << "ValueError: expect x and y to be equal.";
  if (x == 1) {
    // raise InternalError.
    LOG(FATAL) << "InternalError: cannot reach here";
  }
}

TVM_FFI_REGISTER_GLOBAL("testing.ErrorTest").set_body_typed(ErrorTest);

class FrontendTestModuleNode : public runtime::ModuleNode {
 public:
  const char* type_key() const final { return "frontend_test"; }

  static constexpr const char* kAddFunctionName = "__add_function";

  virtual ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self);

 private:
  std::unordered_map<std::string, ffi::Function> functions_;
};

constexpr const char* FrontendTestModuleNode::kAddFunctionName;

ffi::Function FrontendTestModuleNode::GetFunction(const String& name,
                                                  const ObjectPtr<Object>& sptr_to_self) {
  if (name == kAddFunctionName) {
    return ffi::TypedFunction<void(std::string, ffi::Function)>(
        [this, sptr_to_self](std::string func_name, ffi::Function pf) {
          CHECK_NE(func_name, kAddFunctionName)
              << "func_name: cannot be special function " << kAddFunctionName;
          functions_[func_name] = pf;
        });
  }

  auto it = functions_.find(name);
  if (it == functions_.end()) {
    return ffi::Function();
  }

  return it->second;
}

runtime::Module NewFrontendTestModule() {
  auto n = make_object<FrontendTestModuleNode>();
  return runtime::Module(n);
}

TVM_FFI_REGISTER_GLOBAL("testing.FrontendTestModule").set_body_typed(NewFrontendTestModule);

TVM_FFI_REGISTER_GLOBAL("testing.sleep_in_ffi").set_body_typed([](double timeout) {
  std::chrono::duration<int64_t, std::nano> duration(static_cast<int64_t>(timeout * 1e9));
  std::this_thread::sleep_for(duration);
});

TVM_FFI_REGISTER_GLOBAL("testing.ReturnsVariant")
    .set_body_typed([](int x) -> Variant<String, IntImm> {
      if (x % 2 == 0) {
        return IntImm(DataType::Int(64), x / 2);
      } else {
        return String("argument was odd");
      }
    });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsVariant")
    .set_body_typed([](Variant<String, Integer> arg) -> String {
      if (auto opt_str = arg.as<String>()) {
        return opt_str.value()->GetTypeKey();
      } else {
        return arg.get<Integer>()->GetTypeKey();
      }
    });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsBool").set_body_typed([](bool arg) -> bool { return arg; });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsInt").set_body_typed([](int arg) -> int { return arg; });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsObjectRefArray").set_body_typed([](Array<Any> arg) -> Any {
  return arg[0];
});

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsMapReturnsValue")
    .set_body_typed([](Map<Any, Any> map, Any key) -> Any { return map[key]; });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsMapReturnsMap")
    .set_body_typed([](Map<Any, Any> map) -> ObjectRef { return map; });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsPrimExpr").set_body_typed([](PrimExpr expr) -> ObjectRef {
  return expr;
});

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsArrayOfPrimExpr")
    .set_body_typed([](Array<PrimExpr> arr) -> ObjectRef {
      for (ObjectRef item : arr) {
        CHECK(item->IsInstance<PrimExprNode>())
            << "Array contained " << item->GetTypeKey() << " when it should contain PrimExpr";
      }
      return arr;
    });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsArrayOfVariant")
    .set_body_typed([](Array<Variant<ffi::Function, PrimExpr>> arr) -> ObjectRef {
      for (auto item : arr) {
        CHECK(item.as<PrimExpr>() || item.as<ffi::Function>())
            << "Array should contain either PrimExpr or ffi::Function";
      }
      return arr;
    });

TVM_FFI_REGISTER_GLOBAL("testing.AcceptsMapOfPrimExpr")
    .set_body_typed([](Map<ObjectRef, PrimExpr> map) -> ObjectRef {
      for (const auto& kv : map) {
        ObjectRef value = kv.second;
        CHECK(value->IsInstance<PrimExprNode>())
            << "Map contained " << value->GetTypeKey() << " when it should contain PrimExpr";
      }
      return map;
    });

/**
 * Simple event logger that can be used for testing purposes
 */
class TestingEventLogger {
 public:
  struct Entry {
    String event;
    double time_us;
  };

  TestingEventLogger() {
    entries_.reserve(1024);
    start_ = std::chrono::high_resolution_clock::now();
  }

  void Record(String event) {
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

TVM_FFI_REGISTER_GLOBAL("testing.record_event")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      if (args.size() != 0 && args[0].try_cast<String>()) {
        TestingEventLogger::ThreadLocal()->Record(args[0].cast<String>());
      } else {
        TestingEventLogger::ThreadLocal()->Record("X");
      }
    });

TVM_FFI_REGISTER_GLOBAL("testing.reset_events")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      TestingEventLogger::ThreadLocal()->Reset();
    });

TVM_FFI_REGISTER_GLOBAL("testing.dump_events").set_body_typed([]() {
  TestingEventLogger::ThreadLocal()->Dump();
});
}  // namespace tvm
