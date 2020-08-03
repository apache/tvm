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
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/relay/attrs/debug.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/elemwise.h>

#include <vector>

#include "./op_common.h"
#include "./type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(DebugAttrs);

Array<te::Tensor> DebugCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                               const Type& out_type) {
  return Array<te::Tensor>{topi::identity(inputs[0])};
}

RELAY_REGISTER_OP("debug")
    .describe(R"code(Enter the interpreter's debugger.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("program", "Tuple", "The program to execute before debugging.")
    .set_support_level(1)
    .set_attrs_type<DebugAttrs>()
    .add_type_rel("Debug", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<FTVMCompute>("FTVMCompute", DebugCompute);

Expr MakeDebug(Expr expr, String name) {
  auto dattrs = make_object<DebugAttrs>();
  if (name.size() > 0) {
    dattrs->debug_func = EnvFunc::Get(name);
  } else {
    dattrs->debug_func = EnvFunc();
  }
  static const Op& op = Op::Get("debug");
  return Call(op, {expr}, Attrs(dattrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.debug").set_body_typed(MakeDebug);

}  // namespace relay
}  // namespace tvm
