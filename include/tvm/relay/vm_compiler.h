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
 * \file tvm/relay/vm_compiler.h
 * \brief Relay byte code compiler.
 */
#ifndef TVM_RELAY_VM_COMPILER_H_
#define TVM_RELAY_VM_COMPILER_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
namespace vm {

using runtime::vm::VirtualMachine;

bool IsClosure(const Function& func);
Module LambdaLift(const Module& module);
Module InlinePrimitives(const Module& module);

/*! \brief Compile a module into a virtual machine which can be executed. */
VirtualMachine CompileModule(const Module& mod);

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_VM_COMPILER_H_
