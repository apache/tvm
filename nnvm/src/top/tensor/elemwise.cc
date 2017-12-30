/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise.cc
 * \brief Elemenwise operators
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {
// undefined op
NNVM_REGISTER_ELEMWISE_UNARY_OP(__undef__)
.describe(R"code(undefined op.

Used to produce invalide node during optimization.

)code" NNVM_ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(0);

// sigmoid
NNVM_REGISTER_ELEMWISE_UNARY_OP(sigmoid)
.describe(R"code(Computes sigmoid.

.. math::
  Y = 1 / (1 + exp(-X))

)code" NNVM_ADD_FILELINE)
.set_support_level(1)
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
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = n0 + n1
    // grad_0 = grad_y
    // grad_1 = grad_y
    return std::vector<NodeEntry>{ograds[0], ograds[0]};
});

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_sub)
.describe(R"code(Element-wise substraction

)code"  NNVM_ADD_FILELINE)
.set_support_level(1)
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
.describe(R"code(Element-wise multiplication

)code"  NNVM_ADD_FILELINE)
.set_support_level(1)
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

// negative
NNVM_REGISTER_ELEMWISE_UNARY_OP(negative)
.describe(R"code(Elemenwise numeric negative

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
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
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    // y = copy(n0)
    // grad_0 = grad_y
    return std::vector<NodeEntry>{ograds[0]};
});

DMLC_REGISTER_PARAMETER(InitOpParam);

// full
NNVM_REGISTER_INIT_OP(full)
.describe(R"code(Fill array with scalar value

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_INIT_OP(zeros)
.describe(R"code(Fill target with zeros

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_INIT_OP(ones)
.describe(R"code(Fill target with ones

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

// full_like
NNVM_REGISTER_ELEMWISE_UNARY_OP(full_like)
  .describe(R"code(Return an scalar value array with the same shape and type
as the input array

)code"  NNVM_ADD_FILELINE)
.set_support_level(1)
.add_arguments(InitOpParam::__FIELDS__())
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{
      MakeNode("zeros_like", n->attrs.name + "_grad",
               {n->inputs[0]})
    };
});

NNVM_REGISTER_ELEMWISE_UNARY_OP(zeros_like)
.describe(R"code(Return an array of zeros with the same shape and type
as the input array.

)code")
.add_argument("data", "Symbol", "The input")
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{
      MakeNode("zeros_like", n->attrs.name + "_grad",
               {n->inputs[0]})
    };
});

NNVM_REGISTER_ELEMWISE_UNARY_OP(ones_like)
.describe(R"code(Return an array of ones with the same shape and type
as the input array.

)code")
.add_argument("data", "Symbol", "The input")
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{
      MakeNode("zeros_like", n->attrs.name + "_grad",
               {n->inputs[0]})
    };
});

// unary scalar op
DMLC_REGISTER_PARAMETER(ScalarParam);

#define NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(op)                        \
  NNVM_REGISTER_ELEMWISE_UNARY_OP(op)                                   \
  .add_arguments(ScalarParam::__FIELDS__())                             \
  .set_attr_parser(ParamParser<ScalarParam>)                            \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ScalarParam>)


NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__add_scalar__)
.describe(R"code(Tensor add scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{ograds[0]};
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__sub_scalar__)
.describe(R"code(Tensor substract scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{ograds[0]};
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__rsub_scalar__)
.describe(R"code(scalar substract Tensor

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    return std::vector<NodeEntry>{
      MakeNode("negative", n->attrs.name + "_grad_0", {ograds[0]})
    };
});

NNVM_REGISTER_ELEMWISE_BINARY_SCALAR(__mul_scalar__)
.describe(R"code(Tensor multiplies scalar

)code"  NNVM_ADD_FILELINE)
.set_support_level(3)
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


struct ElementWiseSumParam : public dmlc::Parameter<ElementWiseSumParam> {
  int num_args;
  DMLC_DECLARE_PARAMETER(ElementWiseSumParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
      .describe("Number of inputs to be summed.");
  }
};

DMLC_REGISTER_PARAMETER(ElementWiseSumParam);

bool ElementWiseSumShape(const NodeAttrs& attrs,
                         std::vector<TShape> *in_attrs,
                         std::vector<TShape> *out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  return ElemwiseAttr<TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, TShape());
}

bool ElementWiseSumType(const NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  CHECK_EQ(out_attrs->size(), 1);
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

std::vector<NodeEntry> ElementWiseSumGrad(
    const NodePtr& n,
    const std::vector<NodeEntry>& ograds) {
  // identity constraints in the beginning for easier shape inference.
  const Op* copy_op = Op::Get("identity");
  CHECK_EQ(ograds.size(), 1);
  std::vector<NodeEntry> ret;
  NodeEntry n_out{n, 0, 0};
  for (size_t i = 0; i < n->inputs.size(); i++) {
    NodePtr id_node = Node::Create();
    id_node->attrs.op = copy_op;
    id_node->inputs = {ograds[0]};
    ret.push_back(NodeEntry{id_node, 0, 0});
  }
  return ret;
}


NNVM_REGISTER_OP(elemwise_sum)
.describe(R"code(Adds all input arguments element-wise.

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ElementWiseSumParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
  uint32_t ret = dmlc::get<ElementWiseSumParam>(attrs.parsed).num_args;
  return ret;
})
.set_attr<nnvm::FInferShape>("FInferShape", ElementWiseSumShape)
.set_attr<nnvm::FInferType>("FInferType", ElementWiseSumType)
.set_attr<nnvm::FGradient>("FGradient", ElementWiseSumGrad)
.add_argument("args", "Symbol[]", "Positional input arguments");

}  // namespace top
}  // namespace nnvm
