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


#include <gtest/gtest.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/onnx/printer.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/node/reflection.h>

namespace tvm {
namespace relay {

TEST(ProtoBuf, SimpleTest) {
  constexpr const char* kModel = R"(
    #[version = "0.0.5"]
    def @main(%data : Tensor[(1, 304, 128, 128), float32],
             %weight1 : Tensor[(304, 1, 3, 3), float32],
             %bias1 : Tensor[(304), float32],
             %weight2 : Tensor[(256, 304, 1, 1), float32],
             %bias2 : Tensor[(256), float32]) -> Tensor[(1, 256, 128, 128), float32] {
      %0 = nn.conv2d(%data, %weight1, padding=[1, 1, 1, 1], groups=304, channels=304, kernel_size=[3, 3]);
      %1 = nn.bias_add(%0, %bias1);
      %2 = nn.relu(%1);
      %3 = nn.conv2d(%2, %weight2, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
      %4 = nn.bias_add(%3, %bias2);
      nn.relu(%4)
    }
  )";

  IRModule module = ParseModule("string", kModel);
  auto opti = transform::Sequential({transform::InferType()});
  opti(module);
  auto func = module->Lookup("main");
  relay::SaveAsOnnx(func, "simple.onnx");
}
TEST(ProtoBuf, SimpleTestConvWithConstWeight) {
  Expr var =
      relay::Var("input", TensorType({4, 56, 56, 32}, DataType::Float(16)));
  auto weight = relay::Constant(runtime::NDArray::Empty(
      {64, 3, 3, 32}, DataType::Float(16), {kDLCPU, 0}));

  auto attrs_obj = tvm::ReflectionVTable::Global()->CreateObject(
      relay::Conv2DAttrs::_type_key, {{"strides", Array<Integer>({1, 1})},
                                      {"kernel_size", Array<Integer>({3, 3})},
                                      {"data_layout", String("NHWC")},
                                      {"kernel_layout", String("OHWI")}});
  auto conv =
      Call(Op::Get("nn.conv2d"), {var, weight}, Downcast<Attrs>(attrs_obj));
  auto module = IRModule::FromExpr(conv);

  auto opti = transform::Sequential({transform::InferType()});
  module = opti(module);
  auto func = module->Lookup("main");
  LOG_INFO << PrettyPrint(func);
  relay::SaveAsOnnx(func, "simple2.onnx");
}

TEST(ProtoBuf, MultInputOutputTest) {
  Expr x = relay::Var("input0", TensorType({4}, DataType::Float(16)));

  Expr y = relay::Var("input1", TensorType({32}, DataType::Float(16)));
  Expr z = relay::Var("input2", TensorType({64}, DataType::Float(16)));
  auto mesh_grid_inputs = Array<Expr>({x, y, z});

  auto attrs_obj = tvm::ReflectionVTable::Global()->CreateObject(
      relay::MeshgridAttrs::_type_key, {{"indexing", String("ij")}});

  auto tuple_inputs = Tuple(mesh_grid_inputs);

  auto mesh_grid =
      Call(Op::Get("meshgrid"), {tuple_inputs}, Downcast<Attrs>(attrs_obj));

  Array<Expr> relu_calls;

  for (size_t i = 0; i < mesh_grid_inputs.size(); i++) {
    relu_calls.push_back(
        Call(Op::Get("nn.relu"), {TupleGetItem(mesh_grid, i)}));
    relu_calls.push_back(
        Call(Op::Get("nn.relu"), {TupleGetItem(tuple_inputs, i)}));
  }
  auto module = IRModule::FromExpr(Tuple(relu_calls));
  auto opti = transform::Sequential({transform::InferType()});
  module = opti(module);
  auto func = module->Lookup("main");
  LOG_INFO << PrettyPrint(func);
  relay::SaveAsOnnx(func, "multi_input_output.onnx");
}
}  // namespace relay
}  // namespace tvm
