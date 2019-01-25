/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file src/relay/op/device_copy.cc
 * \brief Crossing device data copy operator.
 *
 * The pattern of this operator is registered as kOpaque. Hence, it could be
 * used as "barrier" to avoid fusing operators belonging to differen devices.
 */

#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "type_relations.h"
#include "../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

// relay.device_copy
TVM_REGISTER_NODE_TYPE(DeviceCopyAttrs);

TVM_REGISTER_API("relay.op._make.device_copy")
.set_body_typed<Expr(Expr, int, int)>([](Expr data, int src_dev_type,
                                    int dst_dev_type) {
  auto attrs = make_node<DeviceCopyAttrs>();
  attrs->src_dev_type = src_dev_type;
  attrs->dst_dev_type = dst_dev_type;
  static const Op& op = Op::Get("device_copy");
  return CallNode::make(op, {data}, Attrs(attrs), {});
});

RELAY_REGISTER_OP("device_copy")
.describe(R"code(
Copy data from one tensor to another. The source and destination might be
on different devices.
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_support_level(10)
.add_type_rel("Identity", IdentityRel)
.set_attr<TOpPattern>("TOpPattern", kOpaque)
.set_attr<TOpIsStateful>("TOpIsStateful", false)
.set_attr<FInferCorrectLayout>("FInferCorrectLayout",
                               ElemwiseArbitraryLayout);

}  // namespace relay
}  // namespace tvm
