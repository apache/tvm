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
 * \file src/relay/backend/contrib/libtorch/codegen.cc
 * \brief Implementation of libtorch codegen.
 */

// clang-format off
#include <dlpack/dlpack.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/contrib/libtorch_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"

#include <ATen/DLConvertor.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/torch.h>
// clang-format on

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*! \brief Attributes of a TorchFunction node */
struct TorchFunctionAttrs : public tvm::AttrsNode<TorchFunctionAttrs> {
  std::string serialized_function;
  int64_t len;

  TVM_DECLARE_ATTRS(TorchFunctionAttrs, "relay.attrs.TorchFunctionAttrs") {
    TVM_ATTR_FIELD(serialized_function).set_default("").describe("Function from fn.save(...)");
    TVM_ATTR_FIELD(len).set_default(-1).describe("Function from fn.save(...)");
  }
};

TVM_REGISTER_NODE_TYPE(TorchFunctionAttrs);

bool TorchOpRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  const auto* sfattrs = attrs.as<TorchFunctionAttrs>();
  std::stringstream str(sfattrs->serialized_function);
  torch::jit::Module mod = torch::jit::load(str);

  std::vector<torch::jit::IValue> inputs;
  for (int i = 0; i < num_inputs; i++) {
    auto* ty = types[i].as<TensorTypeNode>();
    ICHECK(ty) << "only accept tensors as inputs";
    std::vector<int64_t> shape;
    for (const auto& s : ty->shape) {
      auto* si = s.as<IntImmNode>();
      if (!si) {
        return false;
      }
      shape.push_back(si->value);
    }
    auto torchScalarType = at::toScalarType(ty->dtype);

    inputs.emplace_back(torch::zeros(shape, at::TensorOptions().dtype(torchScalarType)));
  }
  auto res = mod.forward(inputs);
  auto res_t = res.toTensor();
  ICHECK((int)types.size() == num_inputs + 1) << "only single output supported";
  Array<PrimExpr> res_sizes;
  for (int d = 0; d < res_t.dim(); d++) {
    res_sizes.push_back(IntImm(DataType::Int(32), res_t.size(d)));
  }
  reporter->Assign(types[num_inputs], TensorType(res_sizes, DataType(at::getDLDataType(res_t))));
  return true;
}

RELAY_REGISTER_OP("torch_op")
    .set_support_level(99)
    .add_type_rel("TorchOpRel", TorchOpRel)
    .set_attrs_type<TorchFunctionAttrs>();

Expr MakeTorchOp(Array<Expr> args, const std::string& serialized_function) {
  static const Op& op = Op::Get("torch_op");
  auto attrs = make_object<TorchFunctionAttrs>();
  attrs->serialized_function = serialized_function;
  attrs->len = serialized_function.size();
  return Call(op, args, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.torchop").set_body_typed(MakeTorchOp);

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module TorchCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  ICHECK(func.defined()) << "Input error: expect a Relay function.";
  const auto* call = func->body.as<CallNode>();
  ICHECK(call) << "Expected call node\n";
  const auto* op_node = call->op.as<OpNode>();
  ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();
  const auto op_name = GetRef<Op>(op_node)->name;
  ICHECK(op_name == "torch_op") << "Unsupported op: " << AsText(call->op, false) << "\n";

  const auto* attrs = call->attrs.as<TorchFunctionAttrs>();
  return tvm::runtime::contrib::TorchRuntimeCreate(func_name, attrs->serialized_function);
}

TVM_REGISTER_GLOBAL("relay.ext.torch").set_body_typed(TorchCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
