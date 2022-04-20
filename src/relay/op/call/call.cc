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
 * \file src/relay/op/call/call.cc
 * \brief Operators for calling lowered functions.
 */

#include "./call.h"

#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "../../transforms/infer_layout_utils.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(CallLoweredAttrs);

// call_lowered
bool CallLoweredRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  // Types = [func, call_args, ret_type]
  if (types.size() != 3u) {
    return false;
  }
  const auto* func_type = types[0].as<FuncTypeNode>();
  if (!func_type) {
    return false;
  }

  const auto* tuple_type_node = types[1].as<TupleTypeNode>();
  if (!tuple_type_node) {
    return false;
  }

  // Constraint to ensure function arguments are the same type as the inputs to the function (modulo
  // the Tuple wrapper)
  reporter->Assign(GetRef<TupleType>(tuple_type_node), TupleType(func_type->arg_types, {}));
  // Constraint to ensure the output of call_lowered is the same as the function's return type
  reporter->Assign(types[2], func_type->ret_type);
  return true;
}

const Op& CallLoweredOp() { return Op::Get("call_lowered"); }

Call CallLowered(GlobalVar lowered_func, Array<Expr> args, CallLoweredAttrs call_lowered_attrs,
                 Span span) {
  auto attrs = make_object<CallLoweredAttrs>(std::move(call_lowered_attrs));
  return Call(CallLoweredOp(), {std::move(lowered_func), Tuple(std::move(args))},
              Attrs(std::move(attrs)), /*type_args=*/{}, std::move(span));
}

TVM_REGISTER_GLOBAL("relay.op.call_lowered")
    .set_body_typed([](Expr lowered_func, Array<Expr> args, Attrs attrs, Span span) {
      const auto* lowered_func_node = lowered_func.as<GlobalVarNode>();
      ICHECK(lowered_func_node) << "Function to call should be GlobalVarNode, but got:" << std::endl
                                << PrettyPrint(lowered_func);
      const auto* call_lowered_attrs = attrs.as<CallLoweredAttrs>();
      ICHECK(call_lowered_attrs) << "Expected attributes to be CallLoweredAttrs, but got "
                                 << attrs->GetTypeKey();
      return CallLowered(GetRef<GlobalVar>(lowered_func_node), std::move(args), *call_lowered_attrs,
                         std::move(span));
    });

RELAY_REGISTER_OP("call_lowered")
    .describe(R"code(Invoke an operation compiled by TVM.)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<CallLoweredAttrs>()
    .add_argument("func", "Function", "The lowered function to call.")
    .add_argument("call_args", "Tuple", "The input tensors.")
    .add_type_rel("CallLoweredRel", CallLoweredRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

CallLoweredProps GetCallLoweredProps(const CallNode* call_node) {
  if (call_node->op == CallLoweredOp()) {
    ICHECK(call_node->args.size() == 2) << "Expected call_lowered to have 2 arguments.";
    const auto* function_node = call_node->args[0].as<GlobalVarNode>();
    ICHECK(function_node) << "Expected first arg to call_lowered to be a GlobalVar. ";

    const auto* tuple_args = call_node->args[1].as<TupleNode>();
    ICHECK(tuple_args) << "Expected second arg to call_lowered to be a Tuple of input arguments.";

    ICHECK(call_node->attrs.defined()) << "Expecting call_lowered to have attributes.";
    const auto* call_lowered_attrs = call_node->attrs.as<CallLoweredAttrs>();
    ICHECK(call_lowered_attrs) << "Expected call_lowered op to have CallLoweredAttrs, but found "
                               << call_node->attrs->GetTypeKey();
    // If the call_node has type_args then they are for the polymorphic 'call_lowered' operator
    // itself which expects the function type and argument type as parameters.
    return {GetRef<GlobalVar>(function_node), tuple_args->fields, *call_lowered_attrs};
  }
  return {};
}

Call GetAnyCall(const CallNode* call_node) {
  CallLoweredProps props = GetCallLoweredProps(call_node);
  if (props.lowered_func.defined()) {
    auto call_lowered_attrs = make_object<CallLoweredAttrs>(props.attrs);
    return Call(std::move(props.lowered_func), std::move(props.arguments),
                Attrs(std::move(call_lowered_attrs)),
                /*type_args=*/{}, call_node->span);
  } else {
    return GetRef<Call>(call_node);
  }
}

bool IsReshapeOnly(const CallLoweredProps& props) {
  if (props.attrs.metadata.count("relay_attrs")) {
    auto dict_attrs = Downcast<DictAttrs>(props.attrs.metadata["relay_attrs"]);
    return dict_attrs.HasNonzeroAttr(attr::kReshapeOnly);
  }
  return false;
}

}  // namespace relay
}  // namespace tvm
