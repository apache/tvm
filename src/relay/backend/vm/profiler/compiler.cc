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
 * \file src/relay/backend/vm/profiler/compiler.cc
 * \brief A compiler from relay::Module to the VM byte code.
 */

#include "../../../../runtime/vm/profiler/vm.h"
#include "../compiler.h"

namespace tvm {
namespace relay {
namespace vm {

class VMCompilerDebug : public VMCompiler {
 public:
  VMCompilerDebug() {}
  void InitVM() override { vm_ = std::make_shared<VirtualMachineDebug>(); }
  virtual ~VMCompilerDebug() {}
};

runtime::Module CreateVMCompilerDebug() {
  std::shared_ptr<VMCompilerDebug> exec = std::make_shared<VMCompilerDebug>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay._vm._VMCompilerProfiler")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = CreateVMCompilerDebug();
    });

}  // namespace vm
}  // namespace relay
}  // namespace tvm
