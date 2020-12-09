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

#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ThreefryGenerateAttrs);

bool ThreefryGenerateRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  const ThreefryGenerateAttrs* param = attrs.as<ThreefryGenerateAttrs>();
  ICHECK_EQ(types.size(), 2) << "ThreefryGenerate should have one input and one output";
  const auto* gen = types[0].as<TensorTypeNode>();

  std::vector<IndexExpr> oshape;
  for (auto& x : param->out_shape) {
    oshape.push_back(x);
  }
  // generate returns the next gen and an array of random values
  reporter->Assign(types[1],
                   TupleType({TensorType(gen->shape, gen->dtype), TensorType(oshape, gen->dtype)}));
  return true;
}

Expr MakeThreefryGenerate(Expr gen, Array<Integer> out_shape) {
  auto attrs = make_object<ThreefryGenerateAttrs>();
  attrs->out_shape = out_shape;
  static const Op& op = Op::Get("threefry_generate");
  return Call(op, {gen}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.threefry_generate").set_body_typed(MakeThreefryGenerate);

RELAY_REGISTER_OP("threefry_generate")
    .describe(
        R"doc(Generate an array of random numbers using the Threefry algorithm.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ThreefryGenerateAttrs>()
    .add_argument("gen", "Tensor", "Input generator")
    .add_type_rel("ThreefryGenerate", ThreefryGenerateRel);

bool ThreefrySplitRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2) << "ThreefrySplit should have one input and one output";
  const auto* gen = types[0].as<TensorTypeNode>();
  reporter->Assign(types[1], TupleType({TensorType(gen->shape, gen->dtype),
                                        TensorType(gen->shape, gen->dtype)}));
  return true;
}

Expr MakeThreefrySplit(Expr gen) {
  static const Op& op = Op::Get("threefry_split");
  return Call(op, {gen}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.threefry_split").set_body_typed(MakeThreefrySplit);

RELAY_REGISTER_OP("threefry_split")
    .describe(
        R"doc(Split an array of random numbers using the Threefry algorithm.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("gen", "Tensor", "Input generator")
    .add_type_rel("ThreefrySplit", ThreefrySplitRel);

}  // namespace relay
}  // namespace tvm
