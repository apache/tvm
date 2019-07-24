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
 * \file unary.cc
 * \brief Unary operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/transform.h>
#include <topi/elemwise.h>
#include <topi/transform.h>
#include "../type_relations.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

#define RELAY_UNARY_COMPUTE(FTOPI)                      \
  [] (const Attrs& attrs,                               \
      const Array<Tensor>& inputs,                      \
      const Type& out_type,                             \
      const Target& target) -> Array<Tensor> {          \
    return {FTOPI(inputs[0])};                          \
  }                                                     \


RELAY_REGISTER_UNARY_OP("log")
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::log));


RELAY_REGISTER_UNARY_OP("exp")
.describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   \exp(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::exp));

RELAY_REGISTER_UNARY_OP("sqrt")
.describe(R"code(Returns the sqrt input array, computed element-wise.

.. math::
   sqrt(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::sqrt));

RELAY_REGISTER_UNARY_OP("rsqrt")
.describe(R"code(Returns the rsqrt input array, computed element-wise.

.. math::
   1/sqrt(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::rsqrt));

RELAY_REGISTER_UNARY_OP("zeros_like")
.describe(R"code(Returns an array of zeros, with same type and shape as the input.
)code" TVM_ADD_FILELINE)
.set_support_level(4);

RELAY_REGISTER_UNARY_OP("ones_like")
.describe(R"code(Returns an array of ones, with same type and shape as the input.
)code" TVM_ADD_FILELINE)
.set_support_level(4);

RELAY_REGISTER_UNARY_OP("sigmoid")
.describe(R"code(Returns the sigmoid input array, computed element-wise.

.. math::
   sigmoid(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::sigmoid));


RELAY_REGISTER_UNARY_OP("copy")
.describe(R"code(Copy a tensor.
)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::identity));

// relay.clip
TVM_REGISTER_NODE_TYPE(ClipAttrs);

TVM_REGISTER_API("relay.op._make.clip")
.set_body_typed<Expr(Expr, double, double)>([](Expr a, double a_min, double a_max) {
    auto attrs = make_node<ClipAttrs>();
    attrs->a_min = a_min;
    attrs->a_max = a_max;
    static const Op& op = Op::Get("clip");
  return CallNode::make(op, {a}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("clip")
.describe(R"code(Clip tensor values.
This function takes a tensor, a minimum value `a_min`, and a maximum value `a_max`, and returns a clipped tensor where all values below `a_min` are set to `a_min` and all values above `a_max` are set to `a_max`. `a_min` and `a_max` are cast to the tensor's dtype.
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.add_type_rel("Identity", IdentityRel)
.set_attr<TOpPattern>("TOpPattern", kElemWise)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
.set_attrs_type_key("relay.attrs.ClipAttrs")
.set_support_level(3);


RELAY_REGISTER_UNARY_OP("floor")
.describe(R"code(Returns the floor of input array, computed element-wise.
)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::floor));


RELAY_REGISTER_UNARY_OP("ceil")
.describe(R"code(Returns the ceil of input array, computed element-wise.

.. math::
   ceil(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::ceil));


RELAY_REGISTER_UNARY_OP("trunc")
.describe(R"code(Returns the trunc of input array, computed element-wise.

.. math::
   trunc(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::trunc));

RELAY_REGISTER_UNARY_OP("round")
.describe(R"code(Returns the round of input array, computed element-wise.

.. math::
   round(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::round));

RELAY_REGISTER_UNARY_OP("sign")
.describe(R"code(Returns the sign of input array, computed element-wise.

.. numpy::
   sign(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::sign));


RELAY_REGISTER_UNARY_OP("abs")
.describe(R"code(Returns the abs of input array, computed element-wise.

.. math::
   abs(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::abs));


RELAY_REGISTER_UNARY_OP("tanh")
.describe(R"code(Returns the tanh of input array, computed element-wise.

.. math::
   Y = sinh(X) / cosh(X)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::tanh));


RELAY_REGISTER_UNARY_OP("negative")
.describe(R"code(Returns the numeric negative of input array, computed element-wise.

.. math::
   -(x)

)code" TVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::negative));


RELAY_REGISTER_UNARY_OP("logical_not")
.describe(R"code(Returns the logical inverse of input array, computed element-wise.

.. math::
   ~(x)

)code" TVM_ADD_FILELINE)
.set_support_level(4)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::logical_not));


// shape_of
TVM_REGISTER_NODE_TYPE(ShapeOfAttrs);

bool ShapeOfRel(const Array<Type>& types,
                int num_inputs,
                const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(num_inputs, 1);
  auto tt = types[0].as<TensorTypeNode>();
  CHECK(tt != nullptr);
  const auto* param = attrs.as<ShapeOfAttrs>();
  CHECK(param != nullptr);
  auto vector_out = tvm::Integer(tt->shape.size());
  reporter->Assign(types[1], TensorTypeNode::make({ vector_out }, param->dtype));
  return true;
}

Array<Tensor> ShapeOfCompute(const Attrs& attrs,
                             const Array<Tensor>& inputs,
                             const Type& out_type,
                             const Target& target) {
  CHECK_EQ(inputs.size(), 1);
  const auto* param = attrs.as<ShapeOfAttrs>();
  CHECK(param != nullptr);
  return {topi::shape(inputs[0], param->dtype)};
}

TVM_REGISTER_API("relay.op._make.shape_of")
.set_body_typed<Expr(Expr, DataType)>([](Expr data, DataType dtype) {
  auto attrs = make_node<ShapeOfAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("shape_of");
  return CallNode::make(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("shape_of")
.describe(R"code(Returns a tensor representing the shape of a tensor.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.ShapeOfAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.add_type_rel("ShapeOf", ShapeOfRel)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<TOpPattern>("TOpPattern", kInjective)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               ElemwiseArbitraryLayout)
.set_support_level(10)
.set_attr<FTVMCompute>("FTVMCompute", ShapeOfCompute);


TVM_REGISTER_NODE_TYPE(NdarraySizeAttrs);

bool NdarraySizeRel(const Array<Type>& types,
             int num_inputs,
             const Attrs& attrs,
             const TypeReporter& reporter) {
  CHECK_EQ(num_inputs, 1);
  auto tt = types[0].as<TensorTypeNode>();
  CHECK(tt != nullptr);
  const auto* param = attrs.as<NdarraySizeAttrs>();
  CHECK(param != nullptr);
  reporter->Assign(types[1], TensorTypeNode::make({1}, param->dtype));
  return true;
}

Array<Tensor> NdarraySizeCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  CHECK_EQ(inputs.size(), 1);
  const auto* param = attrs.as<NdarraySizeAttrs>();
  CHECK(param != nullptr);
  return Array<Tensor>{topi::ndarray_size(inputs[0], param->dtype)};
}

TVM_REGISTER_API("relay.op.contrib._make.ndarray_size")
.set_body_typed<Expr(Expr, DataType)>([](Expr data, DataType dtype) {
  auto attrs = make_node<NdarraySizeAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("contrib.ndarray_size");
  return CallNode::make(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("contrib.ndarray_size")
.describe(R"code(Returns a tensor representing the number of elements of input tensor.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_attrs_type_key("relay.attrs.NdarraySizeAttrs")
.add_argument("data", "Tensor", "The input tensor.")
.add_type_rel("NdarraySize", NdarraySizeRel)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<TOpPattern>("TOpPattern", kInjective)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
ElemwiseArbitraryLayout)
.set_support_level(10)
.set_attr<FTVMCompute>("FTVMCompute", NdarraySizeCompute);

}  // namespace relay
}  // namespace tvm
