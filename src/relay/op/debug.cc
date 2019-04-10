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
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/debug.h>
#include <topi/elemwise.h>
#include <vector>
#include "./type_relations.h"
#include "./op_common.h"

namespace tvm {
namespace relay {

Array<Tensor> DebugCompute(const Attrs& attrs,
                               const Array<Tensor>& inputs,
                               const Type& out_type,
                               const Target& target) {
  return Array<Tensor>{ topi::identity(inputs[0]) };
}

RELAY_REGISTER_OP("debug")
.describe(R"code(Enter the interpreter's debugger.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("program", "Tuple", "The program to execute before debugging.")
.set_support_level(1)
.add_type_rel("Debug", IdentityRel)
.set_attr<TOpPattern>("TOpPattern", kOpaque)
.set_attr<FTVMCompute>("FTVMCompute", DebugCompute);

Expr MakeDebug(Expr expr, std::string name) {
  auto dattrs = make_node<DebugAttrs>();
  if (name.size() > 0) {
    dattrs->debug_func = EnvFunc::Get(name);
  } else {
    dattrs->debug_func = EnvFunc();
  }
  static const Op& op = Op::Get("debug");
  return CallNode::make(op, {expr}, Attrs(dattrs), {});
}

TVM_REGISTER_API("relay.op._make.debug")
.set_body_typed(MakeDebug);

}  // namespace relay
}  // namespace tvm

