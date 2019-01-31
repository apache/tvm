/*!
 *  Copyright (c) 2018 by Contributors
 * \file unary.cc
 * \brief Unary operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/transform.h>
#include <topi/elemwise.h>
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
.describe(R"code(Returns the rsqrt input array, computed element-wise.

.. math::
   sqrt(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_UNARY_COMPUTE(topi::sqrt));


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

}  // namespace relay
}  // namespace tvm
