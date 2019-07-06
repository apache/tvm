
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
 *  Copyright (c) 2019 by Contributors
 * \file src/relay/backend/vm/vm.cc
 * \brief The Relay virtual machine.
 */

#include <tvm/relay/interpreter.h>
#include <tvm/logging.h>
#include <tvm/relay/module.h>
#include <tvm/runtime/vm.h>
#include <tvm/relay/analysis.h>

namespace tvm {
namespace relay {
namespace vm {

runtime::vm::VirtualMachine CompileModule(const Module& mod);

using tvm::runtime::Object;
using tvm::runtime::ObjectTag;
using tvm::runtime::vm::VirtualMachine;

VirtualMachine FromModule(const Module& module, const std::vector<TVMContext>& ctxs) {
  auto vm = CompileModule(module);
  vm.Init(ctxs);
  return vm;
}

Object EvaluateModule(const Module& module, const std::vector<TVMContext> ctxs,
                      const std::vector<Object>& vm_args) {
  VirtualMachine vm = FromModule(module, ctxs);
  // TODO(zhiics): This measurement is for temporary usage. Remove it later. We
  // need to introduce a better profiling method.
#if ENABLE_PROFILING
  DLOG(INFO) << "Entry function is main." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
#endif  // ENABLE_PROFILING
  Object res = vm.Invoke("main", vm_args);
#if ENABLE_PROFILING
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  LOG(INFO) << "Inference time: " << duration << "ms\n";
#endif  // ENABLE_PROFILING
  return res;
}

Value VMToValue(const relay::Module& module, Object obj) {
  CHECK(module.defined());
  switch (obj->tag) {
    case ObjectTag::kTensor: {
      return TensorValueNode::make(ToNDArray(obj));
    }
    case ObjectTag::kDatatype: {
      const auto& data_type = obj.AsDatatype();

      tvm::Array<Value> fields;
      for (size_t i = 0; i < data_type->fields.size(); ++i) {
        fields.push_back(VMToValue(module, data_type->fields[i]));
      }

      return ConstructorValueNode::make(data_type->tag, fields);
    }
    default:
      LOG(FATAL) << "unsupported return value of type: " << obj->tag;
      return Value();
  }
}

TVM_REGISTER_API("relay._vm._Tensor").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = Object::Tensor(args[0]);
});

TVM_REGISTER_API("relay._vm._Tuple").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::vector<Object> fields;
  for (auto i = 0; i < args.size(); i++) {
    fields.push_back(args[i]);
  }
  *ret = Object::Tuple(fields);
});

template <typename T>
std::string ToString(const T& t) {
  std::stringstream s;
  s << t;
  return s.str();
}

TVM_REGISTER_API("relay._vm._ObjectTag").set_body([](TVMArgs args, TVMRetValue* ret) {
  Object obj = args[0];
  *ret = ToString(obj->tag);
});

TVM_REGISTER_API("relay._vm._Datatype")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    int itag = args[0];
    size_t tag = static_cast<size_t>(itag);
    std::vector<Object> fields;
    for (int i = 1; i < args.size(); i++) {
      fields.push_back(args[i]);
    }

    *ret = Object::Datatype(tag, fields);
});

TVM_REGISTER_API("relay._vm._evaluate_vm").set_body([](TVMArgs args, TVMRetValue* ret) {
  NodeRef to_compile = args[0];
  TVMContext ctx;
  int dev_type = args[1];
  ctx.device_type = static_cast<DLDeviceType>(dev_type);
  ctx.device_id = args[2];

  Module module;
  if (to_compile.as<FunctionNode>()) {
    Function to_compile = args[0];
    module = ModuleNode::FromExpr(to_compile);
  } else if (to_compile.as<ModuleNode>()) {
    module = args[0];
  } else {
    LOG(FATAL) << "expected function or module";
  }

  std::vector<Object> vm_args;
  for (auto i = 3; i < args.size(); i++) {
    Object obj = args[i];
    vm_args.push_back(obj);
  }

  auto result = EvaluateModule(module, {ctx}, vm_args);
  DLOG(INFO) << "Evaluate VM returning: result=" << result->tag;
  *ret = VMToValue(module, result);
});

}  // namespace vm
}  // namespace relay
}  // namespace tvm
