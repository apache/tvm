/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file src/relay/op/annotation/annotation.cc
 * \brief Registration of annotation operators.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <topi/elemwise.h>

#include "../type_relations.h"
#include "../../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

// relay.annotation.on_device
TVM_REGISTER_NODE_TYPE(OnDeviceAttrs);

TVM_REGISTER_API("relay.op.annotation._make.on_device")
.set_body_typed<Expr(Expr, int)>([](Expr data, int device_type) {
  auto attrs = make_node<OnDeviceAttrs>();
  attrs->device_type = device_type;
  static const Op& op = Op::Get("on_device");
  return CallNode::make(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("on_device")
.describe(R"code(Annotate an expression with device type)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_support_level(10)
.add_type_rel("Identity", IdentityRel)
.set_attr<TOpPattern>("TOpPattern", kOpaque)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               ElemwiseArbitraryLayout);

Expr StopFusion(Expr data) {
  static const Op& op = Op::Get("annotation.stop_fusion");
  return CallNode::make(op, {data}, Attrs{}, {});
}

TVM_REGISTER_API("relay.op.annotation._make.stop_fusion")
.set_body_typed<Expr(Expr)>([](Expr data) {
    return StopFusion(data);
});

RELAY_REGISTER_OP("annotation.stop_fusion")
.describe(R"code(Annotate an expression to prevent it being fused with previous expressions.)code"
TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input data.")
.add_type_rel("Identity", IdentityRel)
.set_support_level(10)
.set_attr<TOpPattern>("TOpPattern", kOpaque)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
.set_attr<FTVMCompute>("FTVMCompute",
                       [](const Attrs& attrs, const Array<Tensor>& inputs,
                          const Type& out_dtype, const Target& target) -> Array<Tensor> {
                         return {topi::identity(inputs[0])};
                       });

}  // namespace relay
}  // namespace tvm
