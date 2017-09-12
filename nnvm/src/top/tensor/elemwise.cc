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
// sigmoid
NNVM_REGISTER_ELEMWISE_UNARY_OP(sigmoid)
.describe(R"code(Computes sigmoid.

.. math::
   y = 1 / (1 + exp(-x))

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// tanh
NNVM_REGISTER_ELEMWISE_UNARY_OP(tanh)
.describe(R"code(Returns the hyperbolic tangent of the input array, computed element-wise.

.. math::
   tanh(x) = sinh(x) / cosh(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// exp
NNVM_REGISTER_ELEMWISE_UNARY_OP(exp)
.describe(R"code(Returns the exp input array, computed element-wise.

.. math::
   exp(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// log
NNVM_REGISTER_ELEMWISE_UNARY_OP(log)
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" NNVM_ADD_FILELINE)
.set_support_level(1);

// binary ops

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_add)
.describe(R"code(Element-wise add

)code")
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_sub)
.describe(R"code(Element-wise substraction

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_mul)
.describe(R"code(Element-wise multiplication

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

NNVM_REGISTER_ELEMWISE_BINARY_OP(elemwise_div)
.describe(R"code(Element-wise multiplication

)code"  NNVM_ADD_FILELINE)
.set_support_level(1);

// negative
NNVM_REGISTER_ELEMWISE_UNARY_OP(negative)
.describe(R"code(Elemenwise numeric negative

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

// copy
NNVM_REGISTER_ELEMWISE_UNARY_OP(copy)
.describe(R"code(Copy tensor to another one.

)code"  NNVM_ADD_FILELINE)
.set_support_level(3);

// unary scalar op
DMLC_REGISTER_PARAMETER(ScalarParam);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__add_scalar__)
.describe(R"code(Tensor add scalar

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__sub_scalar__)
.describe(R"code(Tensor substract scalar

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__rsub_scalar__)
.describe(R"code(scalar substract Tensor

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__mul_scalar__)
.describe(R"code(Tensor multiplies scalar

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__div_scalar__)
.describe(R"code(Tensor divides scalar

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__rdiv_scalar__)
.describe(R"code(scalar divides Tensor

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__pow_scalar__)
.describe(R"code(Tensor power scalar

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);

NNVM_REGISTER_ELEMWISE_UNARY_OP(__rpow_scalar__)
.describe(R"code(scalar power Tensor

)code"  NNVM_ADD_FILELINE)
.set_attr_parser(ParamParser<ScalarParam>)
.set_support_level(3);


}  // namespace top
}  // namespace nnvm
