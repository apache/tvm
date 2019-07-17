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
 * Copyright (c) 2018 by Contributors
 *
 * \file quantize.cc
 *
 * \brief transform a graph to a low-bit graph
 *   for compression and acceleration.
 */
#include <dmlc/thread_local.h>
#include <tvm/base.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <cmath>
#include <string>
#include <vector>
#include <stack>
#include <utility>
#include "pattern_util.h"
#include "quantize.h"


namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

/*! \brief Attribute for simulated quantize operator */
struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  int kind;
  bool sign;
  std::string rounding;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(kind)
        .describe("kind of field, hint for nbit/dtype configuration.");
    TVM_ATTR_FIELD(sign).set_default(true)
        .describe("whether to use signed data type.");
    TVM_ATTR_FIELD(rounding).set_default("round")
        .describe("rounding mode. Can be 'floor', 'ceil', 'round'");
  }
};

TVM_REGISTER_NODE_TYPE(SimulatedQuantizeAttrs);

bool SimulatedQuantizeRel(const Array<Type>& types,
                          int num_inputs,
                          const Attrs& attrs,
                          const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto param = attrs.as<SimulatedQuantizeAttrs>();
  CHECK(param != nullptr);

  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  CHECK_NE(data->shape.size(), 0) << "Input shape cannot be empty";

  reporter->Assign(types[1], TensorTypeNode::make({}, Float(32)));    // dom_scale
  reporter->Assign(types[2], TensorTypeNode::make({}, Float(32)));    // clip_min
  reporter->Assign(types[3], TensorTypeNode::make({}, Float(32)));    // clip_max
  reporter->Assign(types[4], types[0]);                               // output
  return true;
}

RELAY_REGISTER_OP("relay.op.annotation.simulated_quantize")
.describe(R"code(simulated quantize op)code" TVM_ADD_FILELINE)
.set_num_inputs(4)
.add_argument("data", "Tensor", "The input data.")
.add_argument("dom_scale", "Tensor", "The domain scale of input data. It should be a scalar")
.add_argument("clip_min", "Tensor", "lower bound. It should be a scalar")
.add_argument("clip_max", "Tensor", "upper bound. It should be a scalar")
.set_attrs_type_key("relay.attrs.SimulatedQuantizeAttrs")
.set_support_level(11)
.add_type_rel("SimulatedQuantize", SimulatedQuantizeRel);

TVM_REGISTER_API("relay._quantize.simulated_quantize")
.set_body_typed<Expr(Expr, Expr, Expr, Expr, int, bool, std::string)>(
  [](Expr data, Expr dom_scale, Expr clip_min, Expr clip_max,
     int kind, bool sign, std::string rounding) {
    auto attrs = make_node<SimulatedQuantizeAttrs>();
    attrs->kind = kind;
    attrs->sign = sign;
    attrs->rounding = rounding;
    static const Op& op = Op::Get("relay.op.annotation.simulated_quantize");
    return CallNode::make(op, {data, dom_scale, clip_min, clip_max}, Attrs(attrs), {});
  });


// =============
// annotate pass

Expr QAnnotateExprNode::Realize() const {
  const auto& cfg = QConfig::Current();
  if (cfg->store_lowbit_output) {
    // store low bit output back for VTA
    const PackedFunc* f = runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
    return (*f)(this->expr, static_cast<int>(kQInput));
  } else {
    return expr;
  }
}

QAnnotateExpr QAnnotateExprNode::make(Expr expr, QAnnotateKind kind) {
  auto rnode = make_node<QAnnotateExprNode>();
  rnode->expr = expr;
  rnode->kind = kind;
  return QAnnotateExpr(rnode);
}

TVM_REGISTER_API("relay._quantize.make_annotate_expr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = QAnnotateExprNode::make(args[0],
      static_cast<QAnnotateKind>(args[1].operator int()));
  });


// =============
// realize pass

Expr QRealizeIntExprNode::Realize() const {
  const auto& cfg = QConfig::Current();
  Expr data = this->data;
  if (cfg->store_lowbit_output) {
    data = Cast(data, cfg->dtype_input);
  }
  // dequantize
  data = Cast(data, Float(32));
  data = Multiply(data, this->dom_scale);
  return data;
}

QRealizeIntExpr QRealizeIntExprNode::make(Expr data, Expr dom_scale, DataType dtype) {
  NodePtr<QRealizeIntExprNode> n = make_node<QRealizeIntExprNode>();
  n->data = std::move(data);
  n->dom_scale = std::move(dom_scale);
  n->dtype = std::move(dtype);
  return QRealizeIntExpr(n);
}


inline Expr ForwardOp(const Call& ref_call, const Array<Expr>& args) {
  return CallNode::make(ref_call->op,
    args, ref_call->attrs, ref_call->type_args);
}


/* calculate `data * s1 / s2`, use shift if possible */
inline Expr MulAndDiv(Expr data, float s1, float s2) {
  // here we assume the dtype of data is dtype activation
  const QConfig& cfg = QConfig::Current();
  if (s1 == s2) return data;

  float factor = s1 / s2;
  float shift_factor = std::log2(factor);
  CHECK_GT(shift_factor, 0);
  if (static_cast<int>(shift_factor) == shift_factor) {
    return LeftShift(data, MakeConstantScalar(cfg->dtype_activation,
                                              static_cast<int>(shift_factor)));
  } else if (static_cast<int>(factor) == factor) {
    return Multiply(data, MakeConstantScalar(cfg->dtype_activation, factor));
  } else {
    LOG(FATAL) << "fall back to float computation";
    data = Cast(data, Float(32));
    return Multiply(data, MakeConstantScalar(Float(32), factor));
  }
}

Expr QuantizeRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  // do not handle data type cast
  const auto param = ref_call->attrs.as<SimulatedQuantizeAttrs>();
  CHECK_EQ(param->rounding, "round");

  Expr dom_scale = new_args[1];
  Expr clip_min = new_args[2];
  Expr clip_max = new_args[3];

  float dom_scale_imm = GetScalarFromConstant<float>(dom_scale);
  float clip_min_imm = GetScalarFromConstant<float>(clip_min);
  float clip_max_imm = GetScalarFromConstant<float>(clip_max);

  // x * idom_scale = y * odom_scale
  // => y = x * idom_scale / odom_scale
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    // int32->int8
    Expr data = n->data;
    float idom_scale_imm = GetScalarFromConstant<float>(n->dom_scale);
    float odom_scale_imm = GetScalarFromConstant<float>(dom_scale);
    if (idom_scale_imm == odom_scale_imm) {
      // same domain scale, only clip
      data = Clip(data, clip_min_imm, clip_max_imm);
      return QRealizeIntExprNode::make(data, dom_scale, n->dtype);
    }

    float shift_nbit = std::log2(odom_scale_imm / idom_scale_imm);
    CHECK_GT(shift_nbit, 0);
    if (static_cast<int>(shift_nbit) == shift_nbit) {
      // use right shift
      if (cfg->round_for_shift) {
        float round_bias = std::pow(2.0, shift_nbit - 1);
        data = Add(data, MakeConstantScalar(cfg->dtype_activation, static_cast<int>(round_bias)));
      }
      data = RightShift(data, MakeConstantScalar(cfg->dtype_activation,
                                                 static_cast<int>(shift_nbit)));
      data = Clip(data, clip_min_imm, clip_max_imm);
      return QRealizeIntExprNode::make(data, dom_scale, n->dtype);
    } else {
      // float computation
      data = Cast(data, Float(32));
      Expr scaled_data = Multiply(data, Divide(n->dom_scale, dom_scale));
      Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
      return QRealizeIntExprNode::make(round_data, dom_scale, Float(32));
    }
  }

  // quantize from real
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  Expr data = new_args[0];
  Expr scaled_data = Multiply(data, MakeConstantScalar(Float(32), 1 / dom_scale_imm));
  Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
  return QRealizeIntExprNode::make(round_data, dom_scale, Float(32));
}

Expr FoldConstantOpt(const Expr& expr) {
  auto mod = ModuleNode::FromExpr(expr);
  mod = transform::FoldConstant()(mod);
  auto entry_func = mod->Lookup("main");
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

RELAY_REGISTER_OP("relay.op.annotation.simulated_quantize")
.set_attr<FForwardRewrite>("FQRealizeRewrite", QuantizeRealize);


Expr Conv2dRealize(const Call& ref_call,
                   const Array<Expr>& new_args,
                   const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  CHECK(lhs);
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
  CHECK(rhs);

  Expr ldata = lhs->data;
  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<Conv2DAttrs>();
  auto attrs = make_node<Conv2DAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = CallNode::make(ref_call->op,
    {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExprNode::make(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", Conv2dRealize);


Expr DenseRealize(const Call& ref_call,
                  const Array<Expr>& new_args,
                  const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->derived_from<TempExprNode>() || !new_args[1]->derived_from<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();

  Expr ldata = lhs->data;
  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<DenseAttrs>();
  auto attrs = make_node<DenseAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = CallNode::make(ref_call->op,
          {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExprNode::make(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.dense")
.set_attr<FForwardRewrite>("FQRealizeRewrite", DenseRealize);


Expr MulRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    // execute the operation with activation data type.
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;
    Expr rdata = rhs->data;

    DataType dtype = cfg->dtype_activation;
    if (lhs->dtype == Float(32)) {
      ldata = Cast(ldata, dtype);
    } else {
      CHECK_EQ(lhs->dtype, dtype);
    }
    if (rhs->dtype == Float(32)) {
      rdata = Cast(rdata, dtype);
    } else {
      CHECK_EQ(rhs->dtype, dtype);
    }

    Expr ret = ForwardOp(ref_call, {ldata, rdata});
    Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
    Expr dom_scale = FoldConstantOpt(mul);
    return QRealizeIntExprNode::make(ret, dom_scale, dtype);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("multiply")
.set_attr<FForwardRewrite>("FQRealizeRewrite", MulRealize);


float ChooseDomScale(const std::vector<const QRealizeIntExprNode*>& nptrs) {
  if (nptrs.size() == 2) {
    // x = a * s1, y = b * s2
    // x + y = (a * s1 / s2 + b) * s2, if s1 > s2
    //       = (a + b * s2 / s1) * s1, if s2 > s1
    float s1 = GetScalarFromConstant<float>(nptrs[0]->dom_scale);
    float s2 = GetScalarFromConstant<float>(nptrs[1]->dom_scale);
    return s1 > s2 ? s2 : s1;
  } else {
    const QConfig& cfg = QConfig::Current();
    float scale = cfg->global_scale;
    return scale / std::pow(2.0, cfg->nbit_activation - 1);
  }
}


/* \brief Unify the dom scale of arguments */
Array<Expr> UnifyDTypeScale(const Array<Expr>& ref_args, const Array<Expr>& args,
                            DataType* dtype_ptr, Expr* scale_ptr) {
  static const Op& simulated_quantize = Op::Get("relay.op.annotation.simulated_quantize");
  const QConfig& cfg = QConfig::Current();

  std::vector<const QRealizeIntExprNode*> nptrs;
  Array<Expr> ret;
  for (auto arg : args) {
    const auto* nptr = arg.as<QRealizeIntExprNode>();
    CHECK(nptr);
    nptrs.push_back(nptr);
    ret.push_back(nptr->data);
  }

  // unify the data type
  CHECK_EQ(ref_args.size(), args.size());
  DataType dtype;
  if (ret.size() == 2 && nptrs[1]->dtype == cfg->dtype_input) {
    dtype = cfg->dtype_input;
  } else {
    dtype = cfg->dtype_activation;
  }
  for (size_t i = 0; i < ret.size(); ++i) {
    auto ref_arg = ref_args[i].as<CallNode>();
    if (nptrs[i]->dtype != dtype) {
      ret.Set(i, Cast(ret[i], dtype));
    } else if (ref_arg && ref_arg->op.same_as(simulated_quantize) &&
               ref_arg->attrs.as<SimulatedQuantizeAttrs>()->kind == kQInput) {
      auto new_arg = Cast(ret[i], cfg->dtype_input);
      new_arg = StopFusion(new_arg);
      ret.Set(i, Cast(new_arg, dtype));
    }
  }

  // unify the dom_scale
  float s = ChooseDomScale(nptrs);
  Expr dom_scale = MakeConstantScalar(Float(32), s);
  for (size_t i = 0; i < ret.size(); ++i) {
    float cur_s = GetScalarFromConstant<float>(nptrs[i]->dom_scale);
    ret.Set(i, MulAndDiv(ret[i], cur_s, s));
  }

  *dtype_ptr = dtype;
  *scale_ptr = dom_scale;
  return ret;
}

Expr AddRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    Array<Expr> ret_args = UnifyDTypeScale(ref_call->args, new_args, &dtype, &dom_scale);
    Expr ret = ForwardOp(ref_call, ret_args);
    return QRealizeIntExprNode::make(ret, dom_scale, dtype);
  }

  CHECK(!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("add")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AddRealize);

Expr ClipRealize(const Call& ref_call,
                 const Array<Expr>& new_args,
                 const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    const auto ref_attrs = ref_call->attrs.as<ClipAttrs>();
    auto attrs = make_node<ClipAttrs>();
    double dom_scale = GetScalarFromConstant<float>(n->dom_scale);
    attrs->a_min = ref_attrs->a_min / dom_scale;
    attrs->a_max = ref_attrs->a_max / dom_scale;

    Expr ret = CallNode::make(ref_call->op,
      {n->data}, Attrs(attrs), ref_call->type_args);
    return QRealizeIntExprNode::make(ret, n->dom_scale, n->dtype);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("clip")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ClipRealize);


Expr ConcatenateRealize(const Call& ref_call,
                        const Array<Expr>& new_args,
                        const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  CHECK_EQ(ref_call->args.size(), 1);

  const auto* tuple = new_args[0].as<TupleNode>();
  const auto* ref_tuple = ref_call->args[0].as<TupleNode>();
  CHECK(tuple);
  CHECK(ref_tuple);
  const Array<Expr>& arr = tuple->fields;
  const Array<Expr>& ref_arr = ref_tuple->fields;

  if (arr[0].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    Array<Expr> ret_args = UnifyDTypeScale(ref_arr, arr, &dtype, &dom_scale);
    Expr ret = ForwardOp(ref_call, {TupleNode::make(ret_args)});
    return QRealizeIntExprNode::make(ret, dom_scale, dtype);
  } else {
    for (auto arg : new_args) {
      CHECK(!arg->derived_from<TempExprNode>());
    }
    return Expr(nullptr);
  }
}

RELAY_REGISTER_OP("concatenate")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ConcatenateRealize);


/* \brief forward the original operator */
Expr IdentityRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = ForwardOp(ref_call, {n->data});
    return QRealizeIntExprNode::make(ret, n->dom_scale, n->dtype);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.relu")
.set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("strided_slice")
.set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("annotation.stop_fusion")
.set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

/* \brief for unary operators which requantize its input to dtype_nbit */
Expr CastDtypeInputRealize(const Call& ref_call,
                           const Array<Expr>& new_args,
                           const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = Cast(n->data, cfg->dtype_input);
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExprNode::make(ret, n->dom_scale, cfg->dtype_input);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.max_pool2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", CastDtypeInputRealize);


Expr AvgPoolRealize(const Call& ref_call,
                    const Array<Expr>& new_args,
                    const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = n->data;
    if (n->dtype != cfg->dtype_activation) {
      data = Cast(n->data, cfg->dtype_activation);
    }
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExprNode::make(ret, n->dom_scale, cfg->dtype_activation);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.avg_pool2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

Expr ForceCastRealize(const Call& ref_call,
                      const Array<Expr>& new_args,
                      const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = Cast(n->data, cfg->dtype_input);
    return QRealizeIntExprNode::make(ret, n->dom_scale, cfg->dtype_input);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("annotation.force_cast")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ForceCastRealize);

TVM_REGISTER_API("relay._quantize.realize")
.set_body_typed<Expr(Expr)>([](const Expr& e) {
  Expr ret = ForwardRewrite(e, "FQRealizeRewrite", nullptr, nullptr);
  return ret;
});


// =============
// qconfig

QConfig qconfig() {
  return QConfig(make_node<QConfigNode>());
}

/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMQConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  QConfig default_config;

  /*! \brief The current build config context */
  std::stack<QConfig> context_stack;

  TVMQConfigThreadLocalEntry() :
    default_config(qconfig()) {
  }
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMQConfigThreadLocalEntry> TVMQConfigThreadLocalStore;

void QConfig::EnterQConfigScope(const QConfig& build_config) {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.push(build_config);
}

void QConfig::ExitQConfigScope() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.pop();
}

QConfig& QConfig::Current() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }

  return entry->default_config;
}

TVM_REGISTER_NODE_TYPE(QConfigNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<QConfigNode>([](const QConfigNode *op, IRPrinter *p) {
  p->stream << "qconfig(";
  p->stream << "nbit_input=" << op->nbit_input << ", ";
  p->stream << "nbit_weight=" << op->nbit_weight << ", ";
  p->stream << "nbit_activation=" << op->nbit_activation << ", ";
  p->stream << "global_scale=" << op->global_scale << ", ";
  p->stream << "skip_conv_layers==" << op->skip_conv_layers << ", ";
  p->stream << "round_for_shift==" << op->round_for_shift << ", ";
  p->stream << "store_lowbit_output==" << op->store_lowbit_output << ", ";
  p->stream << "debug_enabled_ops==" << op->debug_enabled_ops;
  p->stream << ")";
});

TVM_REGISTER_API("relay._quantize._GetCurrentQConfig")
.set_body_typed(QConfig::Current);

TVM_REGISTER_API("relay._quantize._EnterQConfigScope")
.set_body_typed(QConfig::EnterQConfigScope);

TVM_REGISTER_API("relay._quantize._ExitQConfigScope")
.set_body_typed(QConfig::ExitQConfigScope);

Pass QuantizeAnnotate() {
  std::function<Expr(const Expr&)> fmulti_ref = [](const Expr& e) {
    if (e->derived_from<TempExprNode>()) {
      const auto* n = e.as<QAnnotateExprNode>();
      CHECK(n);
      const PackedFunc* f =
          runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
      Expr ret = (*f)(n->expr, static_cast<int>(kQInput));
      return static_cast<Expr>(QAnnotateExprNode::make(ret, kQInput));
    }
    return e;
  };

  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      auto func = Downcast<Function>(ForwardRewrite(f, "FQAnnotateRewrite", nullptr, fmulti_ref));
      auto new_params = func->params;
      for (const auto& x : FreeVars(func)) {
        new_params.push_back(x);
      }
      return FunctionNode::make(new_params,
                                func->body,
                                func->ret_type,
                                func->type_params,
                                func->attrs);
  };
  return CreateFunctionPass(pass_func, 1, "QuantizeAnnotate", {});
}

TVM_REGISTER_API("relay._quantize.QuantizeAnnotate")
.set_body_typed(QuantizeAnnotate);

Pass QuantizeRealizePass() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(
          ForwardRewrite(f, "FQRealizeRewrite", nullptr, nullptr));
  };
  return CreateFunctionPass(pass_func, 1, "QuantizeRealize", {});
}

TVM_REGISTER_API("relay._quantize.QuantizeRealize")
.set_body_typed(QuantizeRealizePass);

Pass QuantizeRewriteForVTAPass() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(
          ForwardRewrite(f, "FQVTARewrite", nullptr, nullptr));
  };
  return CreateFunctionPass(pass_func, 1, "QuantizeRewriteForVTA", {});
}

TVM_REGISTER_API("relay._quantize.QuantizeRewriteForVTA")
.set_body_typed(QuantizeRewriteForVTAPass);

// =============
// Insert stop_fusion for vta.


Expr QVTAExprNode::Realize() const {
  Expr ret = ForceCast(this->expr);
  return StopFusion(ret);
}

QVTAExpr QVTAExprNode::make(Expr expr) {
  auto rnode = make_node<QVTAExprNode>();
  rnode->expr = expr;
  return QVTAExpr(rnode);
}

TVM_REGISTER_API("relay._quantize.make_vta_expr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = QVTAExprNode::make(args[0]);
  });

TVM_REGISTER_API("relay._quantize.make_stop_fusion")
.set_body_typed<Expr(Expr)>([] (const Expr& expr) {
  return StopFusion(expr);
});

TVM_REGISTER_API("relay._quantize.temp_expr_realize")
.set_body_typed<Expr(Expr)>([] (const Expr& expr) {
  const QVTAExprNode* n = expr.as<QVTAExprNode>();
  CHECK(n);
  return n->Realize();
});


}  // namespace quantize
}  // namespace relay
}  // namespace tvm
