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
 * \file unary.cc
 * \brief Unary operators.
 */
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

#define RELAY_UNARY_COMPUTE(FTOPI)                        \
  [](const Attrs& attrs, const Array<te::Tensor>& inputs, \
     const Type& out_type) -> Array<te::Tensor> { return {FTOPI(inputs[0])}; }

RELAY_REGISTER_UNARY_OP("log")
    .describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::log));

RELAY_REGISTER_UNARY_OP("log2")
    .describe(R"code(Returns the log to base 2 of input array, computed element-wise.

.. math::
   log2(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::log2));

RELAY_REGISTER_UNARY_OP("log10")
    .describe(R"code(Returns the log to base 10 of input array, computed element-wise.

.. math::
   log10(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::log10));

RELAY_REGISTER_UNARY_OP("tan")
    .describe(R"code(Returns the tan of input array, computed element-wise.

.. math::
   Y = tan(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::tan));

RELAY_REGISTER_UNARY_OP("cos")
    .describe(R"code(Returns the cos of input array, computed element-wise.

.. math::
   Y = cos(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::cos));

RELAY_REGISTER_UNARY_OP("cosh")
    .describe(R"code(Returns the cosh of input array, computed element-wise.

.. math::
   Y = cosh(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::cosh));

RELAY_REGISTER_UNARY_OP("sin")
    .describe(R"code(Returns the sin of input array, computed element-wise.

.. math::
   Y = sin(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::sin));

RELAY_REGISTER_UNARY_OP("sinh")
    .describe(R"code(Returns the sinh of input array, computed element-wise.

.. math::
   Y = sinh(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::sinh));

RELAY_REGISTER_UNARY_OP("acos")
    .describe(R"code(Returns the acos of input array, computed element-wise.

.. math::
   Y = acos(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::acos));

RELAY_REGISTER_UNARY_OP("acosh")
    .describe(R"code(Returns the acosh of input array, computed element-wise.

.. math::
   Y = acosh(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::acosh));

RELAY_REGISTER_UNARY_OP("asin")
    .describe(R"code(Returns the asin of input array, computed element-wise.

.. math::
   Y = asin(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::asin));

RELAY_REGISTER_UNARY_OP("asinh")
    .describe(R"code(Returns the asinh of input array, computed element-wise.

.. math::
   Y = asinh(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::asinh));

RELAY_REGISTER_UNARY_OP("atan")
    .describe(R"code(Returns the atan of input array, computed element-wise.

.. math::
   Y = atan(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::atan));

RELAY_REGISTER_UNARY_OP("atanh")
    .describe(R"code(Returns the atanh of input array, computed element-wise.

.. math::
   Y = atanh(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::atanh));

RELAY_REGISTER_UNARY_OP("exp")
    .describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   \exp(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::exp));

RELAY_REGISTER_UNARY_OP("fast_exp")
    .describe(R"code(Returns the fast_exp input array, computed element-wise.

.. math::
   \fast_exp(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::fast_exp));

RELAY_REGISTER_UNARY_OP("erf")
    .describe(R"code(Returns the error function value for input array, computed element-wise.

.. math::
   \erf(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::erf));

RELAY_REGISTER_UNARY_OP("fast_erf")
    .describe(R"code(Returns the error function value for input array, computed element-wise.

.. math::
   \fast_erf(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::fast_erf));

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

Expr MakeClip(Expr a, double a_min, double a_max) {
  auto attrs = make_object<ClipAttrs>();
  attrs->a_min = a_min;
  attrs->a_max = a_max;
  static const Op& op = Op::Get("clip");
  return Call(op, {a}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.clip").set_body_typed(MakeClip);

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
    .set_attrs_type<ClipAttrs>()
    .set_support_level(3);

// relay.fixed_point_multiply
TVM_REGISTER_NODE_TYPE(FixedPointMultiplyAttrs);

TVM_REGISTER_GLOBAL("relay.op._make.fixed_point_multiply")
    .set_body_typed([](Expr a, int32_t multiplier, int32_t shift) {
      auto attrs = make_object<FixedPointMultiplyAttrs>();
      attrs->multiplier = multiplier;
      attrs->shift = shift;
      static const Op& op = Op::Get("fixed_point_multiply");
      return Call(op, {a}, Attrs(attrs), {});
    });

RELAY_REGISTER_OP("fixed_point_multiply")
    .describe(R"code(fixed point multiplication)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attrs_type<FixedPointMultiplyAttrs>()
    .set_support_level(10);

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

RELAY_REGISTER_UNARY_OP("fast_tanh")
    .describe(R"code(Returns the fast_tanh of input array, computed element-wise.

.. math::
   Y = sinh(X) / cosh(X)

)code" TVM_ADD_FILELINE)
    .set_support_level(1)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::fast_tanh));

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
   !(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::logical_not));

RELAY_REGISTER_UNARY_OP("bitwise_not")
    .describe(R"code(Returns the bitwise inverse of input array, computed element-wise.

.. math::
   ~(x)

)code" TVM_ADD_FILELINE)
    .set_support_level(4)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::bitwise_not));

Array<te::Tensor> ShapeOfCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                 const Type& out_type) {
  ICHECK_EQ(inputs.size(), 1);
  const auto* param = attrs.as<ShapeOfAttrs>();
  ICHECK(param != nullptr);
  return {topi::shape(inputs[0], param->dtype)};
}

Expr MakeShapeOf(Expr data, DataType dtype) {
  auto attrs = make_object<ShapeOfAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("shape_of");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.shape_of").set_body_typed(MakeShapeOf);

RELAY_REGISTER_OP("shape_of")
    .describe(R"code(Returns a tensor representing the shape of a tensor.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ShapeOfAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .add_type_rel("ShapeOf", ShapeOfRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    // Use kOpaque for shape_of op for now since it won't be performance critic,
    // and it makes things easier for dynamic shape func
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_support_level(10)
    .set_attr<FTVMCompute>("FTVMCompute", ShapeOfCompute);

TVM_REGISTER_NODE_TYPE(NdarraySizeAttrs);

bool NdarraySizeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 1);
  auto tt = types[0].as<TensorTypeNode>();

  if (tt == nullptr) {
    return false;
  }

  const auto* param = attrs.as<NdarraySizeAttrs>();
  ICHECK(param != nullptr);
  reporter->Assign(types[1], TensorType({}, param->dtype));
  return true;
}

Array<te::Tensor> NdarraySizeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                     const Type& out_type) {
  ICHECK_EQ(inputs.size(), 1);
  const auto* param = attrs.as<NdarraySizeAttrs>();
  ICHECK(param != nullptr);
  return Array<te::Tensor>{topi::ndarray_size(inputs[0], param->dtype)};
}

TVM_REGISTER_GLOBAL("relay.op._make.ndarray_size").set_body_typed([](Expr data, DataType dtype) {
  auto attrs = make_object<NdarraySizeAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("ndarray_size");
  return Call(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("ndarray_size")
    .describe(R"code(Returns a tensor representing the number of elements of input tensor.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<NdarraySizeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .add_type_rel("NdarraySize", NdarraySizeRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TOpPattern>("TOpPattern", kInjective)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_support_level(10)
    .set_attr<FTVMCompute>("FTVMCompute", NdarraySizeCompute);

RELAY_REGISTER_UNARY_OP("isnan")
    .describe(R"code(Returns whether the input contains any NaN, computed element-wise.
.. math::
   isnan(x)
)code" TVM_ADD_FILELINE)
    .set_support_level(3)
    .add_type_rel("IdentityCompRel", IdentityCompRel)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::isnan));

RELAY_REGISTER_UNARY_OP("isfinite")
    .describe(R"code(Returns the finiteness of input, computed element-wise.
.. math::
   isfinite(x)
)code" TVM_ADD_FILELINE)
    .set_support_level(3)
    .add_type_rel("IdentityCompRel", IdentityCompRel)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::isfinite));

RELAY_REGISTER_UNARY_OP("isinf")
    .describe(R"code(Returns the infiniteness of input, computed element-wise.
.. math::
   isinf(x)
)code" TVM_ADD_FILELINE)
    .set_support_level(3)
    .add_type_rel("IdentityCompRel", IdentityCompRel)
    .set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::isinf));

}  // namespace relay
}  // namespace tvm
