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
 * Copyright (c) 2019 by Contributors
 *
 * \file src/relay/pass/print_ir.cc
 *
 * \brief Print the module IR to help debugging.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

namespace transform {

Pass PrintIR() {
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func =
    [=](Module m, PassContext pc) {
      LOG(INFO) << "Dumping the module IR: " << std::endl << AsText(m);
      return m;
  };
  return CreateModulePass(pass_func, 0, "PrintIR", {});
}

TVM_REGISTER_API("relay._transform.PrintIR")
.set_body_typed(PrintIR);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
