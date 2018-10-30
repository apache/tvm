/*!
 * Copyright (c) 2018 by Contributors
 * \file simplify_inference.cc
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include "./pattern_util.h"

namespace tvm {
namespace relay {

// TODO: make type generic
Constant make_const(float x) {
  DLDataType dtype{kDLFloat, 32, 1};
  runtime::NDArray data = runtime::NDArray::Empty({}, dtype, {kDLCPU, 0});
  float* pdata = static_cast<float*>(data->data);
  *pdata = x;
  Constant n = ConstantNode::make(data);
  return n;
}

Expr
BatchNormToInferUnpack(const Attrs attrs,
                       Expr data,
                       Expr gamma,
                       Expr beta,
                       Expr moving_mean,
                       Expr moving_var) {
  const auto param = attrs.as<BatchNormAttrs>();
  Expr epsilon = make_const(param->epsilon);
  Expr var_add_eps = CallNode::make(Op::Get("add"), {moving_var, epsilon});
  Expr sqrt = CallNode::make(Op::Get("sqrt"), {var_add_eps});
  Expr scale = CallNode::make(Op::Get("divide"), {make_const(1.0f), sqrt});

  if (param->scale) {
    scale = CallNode::make(
      Op::Get("multiply"), {scale, gamma});
  }
  Expr neg_mean = CallNode::make(Op::Get("negative"), {moving_mean});
  Expr shift = CallNode::make(Op::Get("multiply"), {neg_mean, scale});
  if (param->center) {
    shift = CallNode::make(Op::Get("add"), {shift, beta});
  }

  int axis = param->axis;
  const auto* tdata = data->type_as<TensorTypeNode>();
  CHECK(tdata) << "require checked type";
  Array<Integer> dshape;
  for (auto e : tdata->shape) {
    CHECK(is_const(e));
    const IntImm* imm = e.as<IntImm>();
    CHECK(imm);
    dshape.push_back(Integer(imm->value));
  }
  scale = ExpandBiasToMatchAxis(scale, axis, dshape);
  shift = ExpandBiasToMatchAxis(shift, axis, dshape);

  Expr out = CallNode::make(Op::Get("multiply"), {data, scale});
  out = CallNode::make(Op::Get("add"), {out, shift});
  return out;
}

class Simplifier : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* n) final {
    if (const OpNode* op = n->op.as<OpNode>()) {
      LOG(INFO) << "op: " << op->name;
      if (op->name == "nn.batch_norm") {
        LOG(INFO) << n->args;
        return BatchNormToInferUnpack(n->attrs, n->args[0], n->args[1], n->args[2], n->args[3], n->args[4]);
      } else if (op->name == "nn.dropout") {
          return n->args[0];
      }
    }
    return GetRef<Expr>(n);
  }
};

Expr SimplifyInference(const Expr& e) {
  return Simplifier().Mutate(e);
}

TVM_REGISTER_API("relay._ir_pass.simplify_inference")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = SimplifyInference(args[0]);
  });

}  // namespace relay
}  // namespace tvm
