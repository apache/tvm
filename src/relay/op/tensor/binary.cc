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
 * \file binary.cc
 * \brief binary broadcast operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/topi/broadcast.h>

#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

#define RELAY_BINARY_COMPUTE(FTOPI)                       \
  [](const Attrs& attrs, const Array<te::Tensor>& inputs, \
     const Type& out_type) -> Array<te::Tensor> {         \
    ICHECK_EQ(inputs.size(), 2U);                         \
    return {FTOPI(inputs[0], inputs[1])};                 \
  }

// Addition
RELAY_REGISTER_BINARY_OP("add")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::add));

// Subtraction
RELAY_REGISTER_BINARY_OP("subtract")
    .describe("Elementwise substract with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::subtract));

// Right shift
RELAY_REGISTER_BINARY_OP("right_shift")
    .describe("Elementwise right shift with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::right_shift));

RELAY_REGISTER_BINARY_OP("left_shift")
    .describe("Elementwise left shift with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::left_shift));

RELAY_REGISTER_BINARY_OP("maximum")
    .describe("Elementwise maximum of two tensors with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::maximum));

RELAY_REGISTER_BINARY_OP("minimum")
    .describe("Elementwise minimum of two tensors with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::minimum));

RELAY_REGISTER_BINARY_OP("divide")
    .describe("Elementwise divide with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::divide));

RELAY_REGISTER_BINARY_OP("floor_divide")
    .describe("Elementwise floor divide with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::floor_divide));

RELAY_REGISTER_BINARY_OP("multiply")
    .describe("Elementwise multiply with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::multiply));

RELAY_REGISTER_BINARY_OP("power")
    .describe("Elementwise power with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::power));

RELAY_REGISTER_BINARY_OP("mod")
    .describe("Elementwise mod with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::mod));

RELAY_REGISTER_BINARY_OP("floor_mod")
    .describe("Elementwise floor mod with broadcasting")
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::floor_mod));

RELAY_REGISTER_BINARY_OP("logical_and")
    .describe("Elementwise logical AND with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::logical_and));

RELAY_REGISTER_BINARY_OP("logical_or")
    .describe("Elementwise logical OR with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::logical_or));

RELAY_REGISTER_BINARY_OP("logical_xor")
    .describe("Elementwise logical XOR with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::logical_xor));

RELAY_REGISTER_BINARY_OP("bitwise_and")
    .describe("Elementwise bitwise AND with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::bitwise_and));

RELAY_REGISTER_BINARY_OP("bitwise_or")
    .describe("Elementwise bitwise OR with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::bitwise_or));

RELAY_REGISTER_BINARY_OP("bitwise_xor")
    .describe("Elementwise bitwise XOR with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::bitwise_xor));

RELAY_REGISTER_CMP_OP("equal")
    .describe("Elementwise equal compare with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::equal));

RELAY_REGISTER_CMP_OP("not_equal")
    .describe("Elementwise not equal with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::not_equal));

RELAY_REGISTER_CMP_OP("less")
    .describe("Elementwise less than with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::less));

RELAY_REGISTER_CMP_OP("less_equal")
    .describe("Elementwise less than or equal compare with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::less_equal));

RELAY_REGISTER_CMP_OP("greater")
    .describe("Elementwise greater than compare with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::greater));

RELAY_REGISTER_CMP_OP("greater_equal")
    .describe("Elementwise greater than or equal compare with broadcasting")
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::greater_equal));

// segment_max
TVM_REGISTER_NODE_TYPE(SegmentAttrs);

bool SegmentRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);

  auto segment_attrs = attrs.as<SegmentAttrs>();
  int num_segments = segment_attrs->num_segments;

  const auto* data = types[0].as<TensorTypeNode>();
  const auto& dshape = data->shape;
  Array<IndexExpr> oshape(dshape);
  oshape.Set(0, num_segments);

  // assign output type
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeSegmentMax(Expr data, Expr segment_ids, int num_segments) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->num_segments = num_segments;
  static const Op& op = Op::Get("segment_max");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.segment_max").set_body_typed(MakeSegmentMax);

RELAY_REGISTER_OP("segment_max")
    .describe(R"doc(Computes the maximum along segments of a tensor.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("segment_ids", "Tensor", "Segments tensor.")
    .set_support_level(5)
    .add_type_rel("SegmentMax", SegmentRel);

Expr MakeSegmentMin(Expr data, Expr segment_ids, int num_segments) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->num_segments = num_segments;
  static const Op& op = Op::Get("segment_min");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.segment_min").set_body_typed(MakeSegmentMin);

RELAY_REGISTER_OP("segment_min")
    .describe(R"doc(Computes the minimum along segments of a tensor.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("segment_ids", "Tensor", "Segments tensor.")
    .set_support_level(5)
    .add_type_rel("SegmentMin", SegmentRel);

Expr MakeSegmentMean(Expr data, Expr segment_ids, int num_segments) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->num_segments = num_segments;
  static const Op& op = Op::Get("segment_mean");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.segment_mean").set_body_typed(MakeSegmentMean);

RELAY_REGISTER_OP("segment_mean")
    .describe(R"doc(Computes the mean along segments of a tensor.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("segment_ids", "Tensor", "Segments tensor.")
    .set_support_level(5)
    .add_type_rel("SegmentMean", SegmentRel);

Expr MakeSegmentSum(Expr data, Expr segment_ids, int num_segments) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->num_segments = num_segments;
  static const Op& op = Op::Get("segment_sum");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.segment_sum").set_body_typed(MakeSegmentSum);

RELAY_REGISTER_OP("segment_sum")
    .describe(R"doc(Computes the sum along segments of a tensor.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("segment_ids", "Tensor", "Segments tensor.")
    .set_support_level(5)
    .add_type_rel("SegmentSum", SegmentRel);

Expr MakeSegmentProd(Expr data, Expr segment_ids, int num_segments) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->num_segments = num_segments;
  static const Op& op = Op::Get("segment_prod");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.segment_prod").set_body_typed(MakeSegmentProd);

RELAY_REGISTER_OP("segment_prod")
    .describe(R"doc(Computes the prod along segments of a tensor.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data.")
    .add_argument("segment_ids", "Tensor", "Segments tensor.")
    .set_support_level(5)
    .add_type_rel("SegmentProd", SegmentRel);

}  // namespace relay
}  // namespace tvm
