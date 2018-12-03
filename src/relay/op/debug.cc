
/*!
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/image.h>
#include <topi/nn.h>
#include <topi/nn/softmax.h>
#include <topi/nn/flatten.h>
#include <vector>
#include "./type_relations.h"
#include "./op_common.h"
#include "./layout.h"

namespace tvm {
namespace relay {

RELAY_REGISTER_OP("debug")
.describe(R"code(Enter the interpreter's debugger.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tuple", "The input list of tensors.")
.set_support_level(1)
.add_type_rel("Debug", IdentityRel)
.set_attr<TNonComputational>("TNonComputational", true)
.set_attr<TOpPattern>("TOpPattern", kInjective);

Expr MakeDebug(Expr expr) {
  static const Op& op = Op::Get("debug");
  return CallNode::make(op, {expr}, Attrs(), {});
}

TVM_REGISTER_API("relay.op._make.debug")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 1>(MakeDebug, args, rv);
  });


}  // namespace relay
}  // namespace tvm

