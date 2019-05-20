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
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Codegen
 * \file c_api_codegen.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/codegen.h>
#include <tvm/lowered_func.h>
#include <tvm/api_registry.h>

namespace tvm {
namespace codegen {

TVM_REGISTER_API("codegen._Build")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<LoweredFunc>()) {
      *ret = Build({args[0]}, args[1]);
    } else {
      *ret = Build(args[0], args[1]);
    }
  });

TVM_REGISTER_API("module._PackImportsToC")
.set_body_typed(PackImportsToC);
}  // namespace codegen
}  // namespace tvm
