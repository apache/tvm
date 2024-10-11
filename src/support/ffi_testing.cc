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
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/runtime/container/variant.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
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

TVM_REGISTER_GLOBAL("testing.nop").set_body([](TVMArgs args, TVMRetValue* ret) {});

TVM_REGISTER_GLOBAL("testing.echo").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = args[0];
});

TVM_REGISTER_GLOBAL("testing.test_wrap_callback").set_body([](TVMArgs args, TVMRetValue* ret) {
  PackedFunc pf = args[0];
  *ret = runtime::TypedPackedFunc<void()>([pf]() { pf(); });
});

TVM_REGISTER_GLOBAL("testing.test_wrap_callback_suppress_err")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      PackedFunc pf = args[0];
      auto result = runtime::TypedPackedFunc<void()>([pf]() {
        try {
          pf();
        } catch (std::exception& err) {
        }
      });
      *ret = result;
    });

TVM_REGISTER_GLOBAL("testing.test_raise_error_callback")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      std::string msg = args[0];
      *ret = runtime::TypedPackedFunc<void()>([msg]() { LOG(FATAL) << msg; });
    });

TVM_REGISTER_GLOBAL("testing.test_check_eq_callback").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string msg = args[0];
  *ret =
      runtime::TypedPackedFunc<void(int x, int y)>([msg](int x, int y) { CHECK_EQ(x, y) << msg; });
});

TVM_REGISTER_GLOBAL("testing.device_test").set_body([](TVMArgs args, TVMRetValue* ret) {
  Device dev = args[0];
  int dtype = args[1];
  int did = args[2];
  CHECK_EQ(static_cast<int>(dev.device_type), dtype);
  CHECK_EQ(static_cast<int>(dev.device_id), did);
  *ret = dev;
});

TVM_REGISTER_GLOBAL("testing.run_check_signal").set_body_typed([](int nsec) {
  for (int i = 0; i < nsec; ++i) {
    tvm::runtime::EnvCheckSignals();
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  LOG(INFO) << "Function finished without catching signal";
});

TVM_REGISTER_GLOBAL("testing.identity_cpp").set_body([](TVMArgs args, TVMRetValue* ret) {
  const auto* identity_func = tvm::runtime::Registry::Get("testing.identity_py");
  ICHECK(identity_func != nullptr)
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

TVM_REGISTER_GLOBAL("testing.ErrorTest").set_body_typed(ErrorTest);

// internal function used for debug and testing purposes
TVM_REGISTER_GLOBAL("testing.object_use_count").set_body([](TVMArgs args, TVMRetValue* ret) {
  runtime::ObjectRef obj = args[0];
  // substract the current one because we always copy
  // and get another value.
  *ret = (obj.use_count() - 1);
});

class FrontendTestModuleNode : public runtime::ModuleNode {
 public:
  const char* type_key() const final { return "frontend_test"; }

  static constexpr const char* kAddFunctionName = "__add_function";

  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self);

 private:
  std::unordered_map<std::string, PackedFunc> functions_;
};

constexpr const char* FrontendTestModuleNode::kAddFunctionName;

PackedFunc FrontendTestModuleNode::GetFunction(const String& name,
                                               const ObjectPtr<Object>& sptr_to_self) {
  if (name == kAddFunctionName) {
    return TypedPackedFunc<void(std::string, PackedFunc)>(
        [this, sptr_to_self](std::string func_name, PackedFunc pf) {
          CHECK_NE(func_name, kAddFunctionName)
              << "func_name: cannot be special function " << kAddFunctionName;
          functions_[func_name] = pf;
        });
  }

  auto it = functions_.find(name);
  if (it == functions_.end()) {
    return PackedFunc();
  }

  return it->second;
}

runtime::Module NewFrontendTestModule() {
  auto n = make_object<FrontendTestModuleNode>();
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("testing.FrontendTestModule").set_body_typed(NewFrontendTestModule);

TVM_REGISTER_GLOBAL("testing.sleep_in_ffi").set_body_typed([](double timeout) {
  std::chrono::duration<int64_t, std::nano> duration(static_cast<int64_t>(timeout * 1e9));
  std::this_thread::sleep_for(duration);
});

TVM_REGISTER_GLOBAL("testing.check_signals").set_body_typed([](double sleep_period) {
  while (true) {
    std::chrono::duration<int64_t, std::nano> duration(static_cast<int64_t>(sleep_period * 1e9));
    std::this_thread::sleep_for(duration);
    runtime::EnvCheckSignals();
  }
});

TVM_REGISTER_GLOBAL("testing.ReturnsVariant").set_body_typed([](int x) -> Variant<String, IntImm> {
  if (x % 2 == 0) {
    return IntImm(DataType::Int(64), x / 2);
  } else {
    return String("argument was odd");
  }
});

TVM_REGISTER_GLOBAL("testing.AcceptsVariant")
    .set_body_typed([](Variant<String, Integer> arg) -> String { return arg->GetTypeKey(); });

TVM_REGISTER_GLOBAL("testing.AcceptsBool").set_body_typed([](bool arg) -> bool { return arg; });

TVM_REGISTER_GLOBAL("testing.AcceptsInt").set_body_typed([](int arg) -> int { return arg; });

TVM_REGISTER_GLOBAL("testing.AcceptsObjectRef").set_body_typed([](ObjectRef arg) -> ObjectRef {
  return arg;
});

TVM_REGISTER_GLOBAL("testing.AcceptsObjectRefArray")
    .set_body_typed([](Array<ObjectRef> arg) -> ObjectRef { return arg[0]; });

TVM_REGISTER_GLOBAL("testing.AcceptsMapReturnsValue")
    .set_body_typed([](Map<ObjectRef, ObjectRef> map, ObjectRef key) -> ObjectRef {
      return map[key];
    });

TVM_REGISTER_GLOBAL("testing.AcceptsMapReturnsMap")
    .set_body_typed([](Map<ObjectRef, ObjectRef> map) -> ObjectRef { return map; });

TVM_REGISTER_GLOBAL("testing.AcceptsPrimExpr").set_body_typed([](PrimExpr expr) -> ObjectRef {
  return expr;
});

TVM_REGISTER_GLOBAL("testing.AcceptsArrayOfPrimExpr")
    .set_body_typed([](Array<PrimExpr> arr) -> ObjectRef {
      for (ObjectRef item : arr) {
        CHECK(item->IsInstance<PrimExprNode>())
            << "Array contained " << item->GetTypeKey() << " when it should contain PrimExpr";
      }
      return arr;
    });

TVM_REGISTER_GLOBAL("testing.AcceptsArrayOfVariant")
    .set_body_typed([](Array<Variant<PackedFunc, PrimExpr>> arr) -> ObjectRef {
      for (ObjectRef item : arr) {
        CHECK(item->IsInstance<PrimExprNode>() || item->IsInstance<runtime::PackedFuncObj>())
            << "Array contained " << item->GetTypeKey()
            << " when it should contain either PrimExpr or PackedFunc";
      }
      return arr;
    });

TVM_REGISTER_GLOBAL("testing.AcceptsMapOfPrimExpr")
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

TVM_REGISTER_GLOBAL("testing.record_event").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args.size() != 0 && args[0].type_code() == kTVMStr) {
    TestingEventLogger::ThreadLocal()->Record(args[0]);
  } else {
    TestingEventLogger::ThreadLocal()->Record("X");
  }
});

TVM_REGISTER_GLOBAL("testing.reset_events").set_body([](TVMArgs args, TVMRetValue* rv) {
  TestingEventLogger::ThreadLocal()->Reset();
});

TVM_REGISTER_GLOBAL("testing.dump_events").set_body_typed([]() {
  TestingEventLogger::ThreadLocal()->Dump();
});
}  // namespace tvm
