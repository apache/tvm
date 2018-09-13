/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise.cc
 * \brief Elemenwise operators
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/tensor.h>
#include <cmath>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/broadcast.h"
#include "topi/elemwise.h"
#include "topi/tags.h"
#include "../../compiler/compile_engine.h"

namespace nnvm {
namespace top {

using namespace tvm;
using namespace nnvm::compiler;

// undefined op
NNVM_REGISTER_ELEMWISE_UNARY_OP(__undef__)
.describe(R"code(undefined op.

Used to produce invalide node during optimization.

)code" NNVM_ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(0);

// floor
NNVM_REGISTER_ELEMWISE_UNARY_OP(floor)
.describe(R"code(Take floor input array, computed element-wise.
)code" NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::floor(inputs[0]) };
});

// ceil
NNVM_REGISTER_ELEMWISE_UNARY_OP(ceil)
.describe(R"code(Take ceil input array, computed element-wise.
)code" NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::ceil(inputs[0]) };
});

// trunc
NNVM_REGISTER_ELEMWISE_UNARY_OP(trunc)
.describe(R"code(Take truncated value of the input, element-wise.
)code" NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::trunc(inputs[0]) };
});

// round
NNVM_REGISTER_ELEMWISE_UNARY_OP(round)
.describe(R"code(Round elements of the input to nearest integer.
)code" NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::round(inputs[0]) };
});

// abs
NNVM_REGISTER_ELEMWISE_UNARY_OP(abs)
.describe(R"code(Take absolute value of elements of the input.
)code" NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::abs(inputs[0]) };
});

// sigmoid
NNVM_REGISTER_ELEMWISE_UNARY_OP(sigmoid)
.describe(R"code(Computes sigmoid.

.. math::
  Y = 1 / (1 + exp(-X))

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::sigmoid(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // y = 1 / (1 + exp(-n0))
    // grad_0 = grad_y * y * (1 - y)
    NodeEntry sub0 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub_0",
                              {ograds[0], NodeEntry{n, 0, 0}});
    NodeEntry sub1 = MakeNode("__rsub_scalar__", n->attrs.name + "_grad_sub_1",
                              {NodeEntry{n, 0, 0}}, {{"scalar", "1"}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad_0",
               {sub0, sub1})
    };
});

// tanh
NNVM_REGISTER_ELEMWISE_UNARY_OP(tanh)
.describe(R"code(Computes hyperbolic tangent.

.. math::
   Y = sinh(X) / cosh(X)

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::tanh(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // y = sinh(n0) / cosh(n0)
    // grad_0 = grad_y * (1 - y^2)
    NodeEntry sub0 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub_0",
                              {NodeEntry{n, 0, 0}, NodeEntry{n, 0, 0}});
    NodeEntry sub1 = MakeNode("__rsub_scalar__", n->attrs.name + "_grad_sub_1",
                              {sub0}, {{"scalar", "1"}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad_0",
               {ograds[0], sub1})
    };
});

// exp
NNVM_REGISTER_ELEMWISE_UNARY_OP(exp)
.describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   exp(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::exp(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // y = exp(n0)
    // grad_0 = grad_y * y
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad_0",
               {ograds[0], NodeEntry{n, 0, 0}})
    };
});

// log
NNVM_REGISTER_ELEMWISE_UNARY_OP(log)
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::log(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // y = log(n0)
    // grad_0 = grad_y / n0
    return std::vector<NodeEntry>{
      MakeNode("elemwise_div", n->attrs.name + "_grad_0",
               {ograds[0], n->inputs[0]})
    };
});

// sqrt
NNVM_REGISTER_ELEMWISE_UNARY_OP(sqrt)
.describe(R"code(Returns the sqrt input array, computed element-wise.

.. math::
   \sqrt(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::sqrt(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    // y = sqrt(n0)
    // grad_0 = grad_y / (2 * y)
    NodeEntry sub0 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_0",
                              {NodeEntry{n, 0, 0}}, {{"scalar", "2"}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_div", n->attrs.name + "_grad_0",
             {ograds[0], sub0})
    };
});

// binary ops

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_add)
.describe(R"code(Element-wise add

)code")
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::add(inputs[0], inputs[1]) };
  })
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0 + n1
    // grad_0 = grad_y
    // grad_1 = grad_y
    return std::vector<NodeEntry>{ MakeNode("copy", n->attrs.name + "_grad_0",
                                            {ograds[0]}),
                                   MakeNode("copy", n->attrs.name + "_grad_0",
                                            {ograds[0]}) };
});

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_sub)
.describe(R"code(Element-wise substraction

)code"  NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ topi::subtract(inputs[0], inputs[1]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0 - n1
    // grad_0 = grad_y
    // grad_1 = - grad_y
    return std::vector<NodeEntry>{
      ograds[0],
      MakeNode("negative", n->attrs.name + "_grad_1", {ograds[0]}),
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mul)
.describe(R"code(Element-wise multiplication

)code"  NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::multiply(inputs[0], inputs[1]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0 * n1
    // grad_0 = grad_y * n1
    // grad_1 = grad_y * n0
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad_0",
               {ograds[0], n->inputs[1]}),
      MakeNode("elemwise_mul", n->attrs.name + "_grad_1",
               {ograds[0], n->inputs[0]})
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_div)
.describe(R"code(Element-wise division

)code"  NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::divide(inputs[0], inputs[1]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0 / n1
    // grad_0 = grad_y / n1
    // grad_1 = - grad_y * n0 / n1^2
    NodeEntry sub0 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub_0",
                              {ograds[0], n->inputs[0]});
    NodeEntry sub1 = MakeNode("negative", n->attrs.name + "_grad_sub_1",
                              {sub0});
    NodeEntry sub2 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub_2",
                              {n->inputs[1], n->inputs[1]});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_div", n->attrs.name + "_grad_0",
               {ograds[0], n->inputs[1]}),
      MakeNode("elemwise_div", n->attrs.name + "_grad_1",
               {sub1, sub2})
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mod)
  .describe(R"code(Element-wise modulo

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::mod(inputs[0], inputs[1]) };
});

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_pow)
  .describe(R"code(Element-wise power

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::power(inputs[0], inputs[1]) };
});

// negative
NNVM_REGISTER_ELEMWISE_UNARY_OP(negative)
.describe(R"code(Elemenwise numeric negative

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::negative(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = - n0
    // grad_0 = - grad_y
    return std::vector<NodeEntry>{
      MakeNode("negative", n->attrs.name + "_grad_0", {ograds[0]}),
    };
});

// copy
NNVM_REGISTER_ELEMWISE_UNARY_OP(copy)
.describe(R"code(Copy tensor to another one.

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
      return Array<Tensor>{ topi::identity(inputs[0]) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = copy(n0)
    // grad_0 = grad_y
    return std::vector<NodeEntry>{ MakeNode("copy", n->attrs.name + "_grad_0",
                                            {ograds[0]}) };
});

DMLC_REGISTER_PARAMETER(InitOpParam);
DMLC_REGISTER_PARAMETER(InitOpWithScalarParam);
DMLC_REGISTER_PARAMETER(FillValueParam);

// full
NNVM_REGISTER_INIT_OP(full)
.describe(R"code(Fill array with scalar value

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<InitOpWithScalarParam>)
.set_attr<FGetAttrDict>(
  "FGetAttrDict", ParamGetAttrDict<InitOpWithScalarParam>)
.add_arguments(InitOpWithScalarParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", ZeroShape<InitOpWithScalarParam>)
.set_attr<FInferType>("FInferType", ZeroType<InitOpWithScalarParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const InitOpWithScalarParam& param = nnvm::get<InitOpWithScalarParam>(attrs.parsed);
    Array<Expr> shape = ShapeToArray(param.shape);
    Type dtype = GetTVMType(param.dtype);
    Expr fill_value = tvm::make_const(dtype, param.fill_value);
    return Array<Tensor>{ topi::full(shape, dtype, fill_value) };
})
.set_support_level(4);

NNVM_REGISTER_INIT_OP(zeros)
.describe(R"code(Fill target with zeros

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<FGetAttrDict>(
  "FGetAttrDict", ParamGetAttrDict<InitOpParam>)
.add_arguments(InitOpParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", ZeroShape<InitOpParam>)
.set_attr<FInferType>("FInferType", ZeroType<InitOpParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const InitOpParam& param = nnvm::get<InitOpParam>(attrs.parsed);
    Array<Expr> shape = ShapeToArray(param.shape);
    Type dtype = GetTVMType(param.dtype);
    Expr fill_value = tvm::make_const(dtype, 0);
    return Array<Tensor>{ topi::full(shape, dtype, fill_value) };
})
.set_support_level(4);

NNVM_REGISTER_INIT_OP(ones)
.describe(R"code(Fill target with ones

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<FGetAttrDict>(
  "FGetAttrDict", ParamGetAttrDict<InitOpParam>)
.add_arguments(InitOpParam::__FIELDS__())
.set_attr<FInferShape>("FInferShape", ZeroShape<InitOpParam>)
.set_attr<FInferType>("FInferType", ZeroType<InitOpParam>)
.set_attr<FCorrectLayout>("FCorrectLayout", ZeroLayout)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const InitOpParam& param = nnvm::get<InitOpParam>(attrs.parsed);
    Array<Expr> shape = ShapeToArray(param.shape);
    Type dtype = GetTVMType(param.dtype);
    Expr fill_value = tvm::make_const(dtype, 1);
    return Array<Tensor>{ topi::full(shape, dtype, fill_value) };
})
.set_support_level(4);

// full_like
NNVM_REGISTER_INIT_LIKE_OP(full_like)
.describe(R"code(Return an scalar value array with the same shape and type
as the input array

)code"  NNVM_ADD_FILELINE)
.add_arguments(FillValueParam::__FIELDS__())
.set_attr_parser(ParamParser<FillValueParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<FillValueParam>)
.set_attr<FTVMCompute>(
    "FTVMCompute", [](const NodeAttrs& attrs,
                      const Array<Tensor>& inputs,
                      const Array<Tensor>& out_info) {
      const FillValueParam& param = nnvm::get<FillValueParam>(attrs.parsed);
      const Expr fill_value = tvm::make_const(out_info[0]->dtype, param.fill_value);
      return Array<Tensor> { topi::full_like(inputs[0], fill_value) };
})
.set_support_level(4);

NNVM_REGISTER_INIT_LIKE_OP(zeros_like)
.describe(R"code(Return an array of zeros with the same shape and type
as the input array.

)code")
.set_attr<FTVMCompute>(
    "FTVMCompute", [](const NodeAttrs& attrs,
                      const Array<Tensor>& inputs,
                      const Array<Tensor>& out_info) {
      return Array<Tensor> { topi::full_like(inputs[0],
                                             tvm::make_const(out_info[0]->dtype, 0)) };
})
.set_support_level(4);

NNVM_REGISTER_INIT_LIKE_OP(ones_like)
.describe(R"code(Return an array of ones with the same shape and type
as the input array.

)code")
.set_attr<FTVMCompute>(
    "FTVMCompute", [](const NodeAttrs& attrs,
                      const Array<Tensor>& inputs,
                      const Array<Tensor>& out_info) {
      return Array<Tensor> { topi::full_like(inputs[0],
                                             tvm::make_const(out_info[0]->dtype, 1)) };
})
.set_support_level(4);

// unary scalar op
DMLC_REGISTER_PARAMETER(ScalarParam);

#define NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(op)                        \
  NNVM_REGISTER_ELEMWISE_UNARY_OP(op)                                   \
  .add_arguments(ScalarParam::__FIELDS__())                             \
  .set_attr_parser(ParamParser<ScalarParam>)                            \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ScalarParam>)

inline Tensor binary_scalar_op(const NodeAttrs& attrs,
                               const Tensor& x,
                               std::function<Expr(Expr, Expr)> f) {
  const ScalarParam& param = nnvm::get<ScalarParam>(attrs.parsed);
  auto scalar_val = static_cast<float>(param.scalar);
  return compute(x->shape, [&](const Array<Var>& i) {
    auto scalar_const = make_const(x->dtype, scalar_val);
    return f(x(i), scalar_const);
    }, "tensor", topi::kElementWise);
}

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__add_scalar__)
.describe(R"code(Tensor add scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return x + y; }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{ MakeNode("copy", n->attrs.name + "_grad_0",
                                            {ograds[0]}) };
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__sub_scalar__)
.describe(R"code(Tensor substract scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return x - y; }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{ograds[0]};
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rsub_scalar__)
.describe(R"code(scalar substract Tensor

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return y - x; }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{
      MakeNode("negative", n->attrs.name + "_grad_0", {ograds[0]})
    };
});


NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__lshift_scalar__)
.describe(R"code(Tensor left shift by scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ScalarParam& param = nnvm::get<ScalarParam>(attrs.parsed);
    int scalar_val = static_cast<int>(param.scalar);
    return Array<Tensor>{
      topi::left_shift(inputs[0],
                       make_const(inputs[0]->dtype, scalar_val))};
    });

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rshift_scalar__)
.describe(R"code(Tensor right shift by scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ScalarParam& param = nnvm::get<ScalarParam>(attrs.parsed);
    int scalar_val = static_cast<int>(param.scalar);
    return Array<Tensor>{
      topi::right_shift(inputs[0],
                        make_const(inputs[0]->dtype, scalar_val))};
  });

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__mul_scalar__)
.describe(R"code(Tensor multiplies scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return x * y; }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0 * scalar
    // grad_0 = grad_y * scalar
    return std::vector<NodeEntry>{
      MakeNode("__mul_scalar__", n->attrs.name + "_grad_0",
               {ograds[0]}, {{"scalar", n->attrs.dict["scalar"]}})
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__div_scalar__)
.describe(R"code(Tensor divides scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return x / y; }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0 / scalar
    // grad_0 = grad_y / scalar
    return std::vector<NodeEntry>{
      MakeNode("__div_scalar__", n->attrs.name + "_grad_0",
               {ograds[0]}, {{"scalar", n->attrs.dict["scalar"]}})
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rdiv_scalar__)
.describe(R"code(scalar divides Tensor

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return y / x; }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = scalar / n0
    // grad_0 = - grad_y * scalar / n0^2
    NodeEntry sub0 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_0",
                              {ograds[0]},
                              {{"scalar", n->attrs.dict["scalar"]}});
    NodeEntry sub1 = MakeNode("negative", n->attrs.name + "_grad_sub_1",
                              {sub0});
    NodeEntry sub2 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub_2",
                              {n->inputs[0], n->inputs[0]});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_div", n->attrs.name + "_grad_0",
               {sub1, sub2})
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__pow_scalar__)
.describe(R"code(Tensor power scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return tvm::pow(x, y); }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0^scalar
    // grad_0 = grad_y * scalar * n0^(scalar - 1)
    double scalar = std::stod(n->attrs.dict["scalar"]);
    NodeEntry sub0 = MakeNode("__pow_scalar__", n->attrs.name + "_grad_sub_0",
                              {n->inputs[0]},
                              {{"scalar", std::to_string(scalar - 1)}});
    NodeEntry sub1 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_1",
                              {ograds[0]},
                              {{"scalar", std::to_string(scalar)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad_0",
               {sub0, sub1})
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rpow_scalar__)
.describe(R"code(scalar power Tensor

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ binary_scalar_op(attrs, inputs[0],
      [](Expr x, Expr y) { return tvm::pow(y, x); }) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = scalar^n0
    // grad_0 = grad_y * scalar^n0 * log(scalar)
    double num = std::stod(n->attrs.dict["scalar"]);
    NodeEntry sub0 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_0",
                              {NodeEntry{n, 0, 0}},
                              {{"scalar", std::to_string(std::log(num))}});
    return std::vector<NodeEntry>{
      MakeNode("__mul_symbol__", n->attrs.name + "_grad_0",
               {ograds[0], sub0})
    };
});

DMLC_REGISTER_PARAMETER(ElementWiseReduceParam);

NNVM_REGISTER_ELEMWISE_REDUCE_OP(elemwise_sum)
.describe(R"code(Adds all input arguments element-wise.

)code"  NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ElementWiseReduceParam& param = nnvm::get<ElementWiseReduceParam>(attrs.parsed);
    CHECK_EQ(param.num_args, inputs.size()) << """Compute definition of elemwise sum""";
    return Array<Tensor>{ topi::elemwise_sum(inputs) };
})
.set_attr<nnvm::FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    CHECK_EQ(ograds.size(), 1);
    std::vector<NodeEntry> ret;
    for (size_t i = 0; i < n->inputs.size(); i++) {
      ret.push_back(MakeNode("copy", n->attrs.name + "_grad_0", {ograds[0]}));
    }
    return ret;
  })
.set_support_level(4);

NNVM_REGISTER_ELEMWISE_UNARY_OP(block_grad)
.describe(R"code(Blocks gradient computation for input.

)code" NNVM_ADD_FILELINE)
.set_attr<nnvm::FInplaceIdentity>(
  "FInplaceIdentity", [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
})
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_support_level(4);

DMLC_REGISTER_PARAMETER(IndicatorParam);

// indicator function
NNVM_REGISTER_INDICATOR_OP(greater)
.describe(R"code(Greater function that returns a mask tensor
with 1.0 if (left > right), otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("lhs", "Tensor", "First input")
.add_argument("rhs", "Tensor", "Second input")
.set_num_inputs(2)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ topi::cast(topi::greater(inputs[0], inputs[1]), out_info[0]->dtype) };
})
.set_support_level(4);


NNVM_REGISTER_INDICATOR_OP(less)
  .describe(R"code(Less function that returns a mask tensor
with 1.0 if (left < right), otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("lhs", "Tensor", "First input")
.add_argument("rhs", "Tensor", "Second input")
.set_num_inputs(2)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ topi::cast(topi::less(inputs[0], inputs[1]), out_info[0]->dtype) };
})
.set_support_level(4);

NNVM_REGISTER_INDICATOR_OP(_max_mask)
  .describe(R"code(Function that returns a mask tensor
with 1.0 if the value is maximum over given axes, otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input")
.set_num_inputs(1)
.add_arguments(IndicatorParam::__FIELDS__())
.set_attr_parser(ParamParser<IndicatorParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<IndicatorParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_support_level(1);

NNVM_REGISTER_INDICATOR_OP(_min_mask)
  .describe(R"code(Function that returns a mask tensor
with 1.0 if the value is minimum over given axes, otherwise 0.0 element-wise.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input")
.set_num_inputs(1)
.add_arguments(IndicatorParam::__FIELDS__())
.set_attr_parser(ParamParser<IndicatorParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<IndicatorParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_support_level(1);


DMLC_REGISTER_PARAMETER(ClipParam);

NNVM_REGISTER_OP(clip)
.describe(R"doc(Clips (limits) the values in an array.
Given an interval, values outside the interval are clipped to the interval edges.
Clipping ``x`` between `a_min` and `a_x` would be::
   clip(x, a_min, a_max) = max(min(x, a_max), a_min))
Example::
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
)doc" NNVM_ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ClipParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ClipParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ClipParam params = get<ClipParam>(attrs.parsed);
    return Array<Tensor>{
      topi::clip(inputs[0], tvm::make_const(tvm::Float(32), params.a_min),
                 tvm::make_const(tvm::Float(32), params.a_max)) };
  })
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(ClipParam::__FIELDS__())
.set_attr<nnvm::FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = clip(x, a_min, a_max)
    // min_mask = greater_equal(x, a_min*ones_like(x))
    //          => ones_like(x) - less(x, a_min)
    // max_mask = less_equal(x, a_max*ones_like(x))
    //          => ones_like(x) - greater(x, a_max)
    // grad_x = min_mask * max_mask * grad_y
    CHECK_EQ(ograds.size(), 1);

    NodeEntry sub0 = MakeNode("ones_like", n->attrs.name + "_grad_sub_0",
                              {n->inputs[0]});
    // min_mask
    NodeEntry sub1 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_1",
                              {sub0}, {{"scalar", n->attrs.dict["a_min"]}});
    NodeEntry sub2 = MakeNode("less", n->attrs.name + "_grad_sub_2",
                              {n->inputs[0], sub1});
    NodeEntry sub3 = MakeNode("elemwise_sub", n->attrs.name + "_grad_sub_3",
                              {sub0, sub2});

    // max_mask
    NodeEntry sub4 = MakeNode("__mul_scalar__", n->attrs.name + "_grad_sub_4",
                              {sub0}, {{"scalar", n->attrs.dict["a_max"]}});
    NodeEntry sub5 = MakeNode("greater", n->attrs.name + "_grad_sub_5",
                              {n->inputs[0], sub4});
    NodeEntry sub6 = MakeNode("elemwise_sub", n->attrs.name + "_grad_sub_6",
                              {sub0, sub5});

    // min_mask * max_mask
    NodeEntry sub7 = MakeNode("elemwise_mul", n->attrs.name + "_grad_sub_7",
                              {sub3, sub6});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad",
               {sub7, ograds[0]})
    };
  })
.set_support_level(4);

}  // namespace top
}  // namespace nnvm
