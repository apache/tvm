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
 *
 * \file realize.cc
 *
 * \brief Realizing the simulated graph into real low-precision
 *   graph.
 */

#include <tvm/relay/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include "./quantize.h"
#include "../transforms/pattern_util.h"
#include "../qnn/util.h"

namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

class QRealizeExpr;
class QRealizeIntExpr;

class QRealizeExprNode : public TempExprNode {
 public:
  Expr data;
  static constexpr const char* _type_key = "relay.quantize.QRealizeExpr";
  TVM_DECLARE_BASE_OBJECT_INFO(QRealizeExprNode, TempExprNode);
};

class QRealizeExpr : public TempExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(QRealizeExpr, TempExpr, QRealizeExprNode);
};


class QRealizeIntExprNode : public QRealizeExprNode {
 public:
  Expr dom_scale;
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("dom_scale", &dom_scale);
    v->Visit("dtype", &dtype);
  }

  Expr Realize() const final;

  static constexpr const char * _type_key = "relay.quantize.QRealizeIntExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(QRealizeIntExprNode, QRealizeExprNode);
};

class QRealizeIntExpr : public QRealizeExpr {
 public:
  TVM_DLL QRealizeIntExpr(Expr data, Expr dom_scale, DataType dtype);

  TVM_DEFINE_OBJECT_REF_METHODS(QRealizeIntExpr, QRealizeExpr, QRealizeIntExprNode);
};


Expr QRealizeIntExprNode::Realize() const {
  Expr data = this->data;
  // dequantize
  data = Cast(data, DataType::Float(32));
  data = Multiply(data, this->dom_scale);
  return data;
}

QRealizeIntExpr::QRealizeIntExpr(Expr data, Expr dom_scale, DataType dtype) {
  ObjectPtr<QRealizeIntExprNode> n = make_object<QRealizeIntExprNode>();
  n->data = std::move(data);
  n->dom_scale = std::move(dom_scale);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}


inline Expr ForwardOp(const Call& ref_call, const Array<Expr>& args) {
  return Call(ref_call->op, args, ref_call->attrs, ref_call->type_args);
}


/* calculate `data * s1 / s2`, use shift if possible */
inline Expr MulAndDiv(Expr data, float s1, float s2, DataType dtype,
                      const Array<IndexExpr> &data_shape) {
  const QConfig& cfg = QConfig::Current();
  // here we assume the dtype of data is dtype activation
  if (s1 == s2) return data;

  float factor = s1 / s2;
  float shift_factor = std::log2(factor);
  CHECK_GT(shift_factor, 0);
  if (static_cast<int>(shift_factor) == shift_factor) {
    return LeftShift(data, MakeConstantScalar(dtype,
                                              static_cast<int>(shift_factor)));
  } else if (static_cast<int>(factor) == factor) {
    return Multiply(data, MakeConstantScalar(dtype, factor));
  } else {
    data = qnn::FixedPointMultiply(data, factor, data_shape, cfg->rounding);
    return Cast(data, dtype);
  }
}

Expr QuantizeRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const ObjectRef& ctx) {
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
      return QRealizeIntExpr(data, dom_scale, n->dtype);
    }

    float shift_nbit = std::log2(odom_scale_imm / idom_scale_imm);
    CHECK_NE(shift_nbit, 0);
    if (static_cast<int>(shift_nbit) == shift_nbit) {
      if (shift_nbit > 0) {
        // use right shift
        if (cfg->round_for_shift) {
          float round_bias = std::pow(2.0, shift_nbit - 1);
          data = Add(data, MakeConstantScalar(cfg->dtype_activation,
                                              static_cast<int>(round_bias)));
        }
        data = RightShift(data, MakeConstantScalar(cfg->dtype_activation,
                                                   static_cast<int>(shift_nbit)));
      } else {
        data = LeftShift(data, MakeConstantScalar(cfg->dtype_activation,
                                                  static_cast<int>(shift_nbit)));
      }
      data = Clip(data, clip_min_imm, clip_max_imm);
      return QRealizeIntExpr(data, dom_scale, n->dtype);
    } else {
      data = Cast(data, DataType::Int(64));
      data = qnn::FixedPointMultiply(data, idom_scale_imm / odom_scale_imm,
                                     ref_call->type_as<TensorTypeNode>()->shape,
                                     cfg->rounding);
      data = Cast(Clip(data, clip_min_imm, clip_max_imm), n->dtype);
      return QRealizeIntExpr(data, dom_scale, n->dtype);
    }
  }

  // quantize from real
  CHECK(!new_args[0]->IsInstance<TempExprNode>());
  Expr data = new_args[0];
  Expr scaled_data = Multiply(data, MakeConstantScalar(DataType::Float(32), 1 / dom_scale_imm));
  Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
  return QRealizeIntExpr(round_data, dom_scale, DataType::Float(32));
}

Expr FoldConstantOpt(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::FoldConstant()(mod);
  auto entry_func = Downcast<Function>(mod->Lookup("main"));
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

RELAY_REGISTER_OP("relay.op.annotation.simulated_quantize")
.set_attr<FForwardRewrite>("FQRealizeRewrite", QuantizeRealize);


Expr Conv2dRealize(const Call& ref_call,
                   const Array<Expr>& new_args,
                   const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() && !new_args[1]->IsInstance<TempExprNode>()) {
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
  auto attrs = make_object<Conv2DAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = Call(ref_call->op,
    {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExpr(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", Conv2dRealize);


Expr DenseRealize(const Call& ref_call,
                  const Array<Expr>& new_args,
                  const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>()) {
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
  auto attrs = make_object<DenseAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = Call(ref_call->op,
          {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExpr(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.dense")
.set_attr<FForwardRewrite>("FQRealizeRewrite", DenseRealize);


Expr MulRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    // execute the operation with activation data type.
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;
    Expr rdata = rhs->data;

    DataType dtype = cfg->dtype_activation;
    if (lhs->dtype != dtype) {
      ldata = Cast(ldata, dtype);
    }
    if (rhs->dtype != dtype) {
      rdata = Cast(rdata, dtype);
    }

    Expr ret = ForwardOp(ref_call, {ldata, rdata});
    Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
    Expr dom_scale = FoldConstantOpt(mul);
    return QRealizeIntExpr(ret, dom_scale, dtype);
  }
  CHECK(!new_args[0]->IsInstance<TempExprNode>() && !new_args[1]->IsInstance<TempExprNode>());
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
  Expr dom_scale = MakeConstantScalar(DataType::Float(32), s);
  for (size_t i = 0; i < ret.size(); ++i) {
    float cur_s = GetScalarFromConstant<float>(nptrs[i]->dom_scale);
    ret.Set(i, MulAndDiv(ret[i], cur_s, s, dtype, ref_args[i]->type_as<TensorTypeNode>()->shape));
  }

  *dtype_ptr = dtype;
  *scale_ptr = dom_scale;
  return ret;
}

Expr AddRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const ObjectRef& ctx) {
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    Array<Expr> ret_args = UnifyDTypeScale(ref_call->args, new_args, &dtype, &dom_scale);
    Expr ret = ForwardOp(ref_call, ret_args);
    return QRealizeIntExpr(ret, dom_scale, dtype);
  }

  CHECK(!new_args[0]->IsInstance<TempExprNode>() && !new_args[1]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("add")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AddRealize);

Expr ClipRealize(const Call& ref_call,
                 const Array<Expr>& new_args,
                 const ObjectRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    const auto ref_attrs = ref_call->attrs.as<ClipAttrs>();
    auto attrs = make_object<ClipAttrs>();
    double dom_scale = GetScalarFromConstant<float>(n->dom_scale);
    attrs->a_min = ref_attrs->a_min / dom_scale;
    attrs->a_max = ref_attrs->a_max / dom_scale;

    Expr ret = Call(ref_call->op,
      {n->data}, Attrs(attrs), ref_call->type_args);
    return QRealizeIntExpr(ret, n->dom_scale, n->dtype);
  }
  CHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("clip")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ClipRealize);


Expr ConcatenateRealize(const Call& ref_call,
                        const Array<Expr>& new_args,
                        const ObjectRef& ctx) {
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
    Expr ret = ForwardOp(ref_call, {Tuple(ret_args)});
    return QRealizeIntExpr(ret, dom_scale, dtype);
  } else {
    for (auto arg : new_args) {
      CHECK(!arg->IsInstance<TempExprNode>());
    }
    return Expr(nullptr);
  }
}

RELAY_REGISTER_OP("concatenate")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ConcatenateRealize);


/* \brief forward the original operator */
Expr IdentityRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const ObjectRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = ForwardOp(ref_call, {n->data});
    return QRealizeIntExpr(ret, n->dom_scale, n->dtype);
  }
  CHECK(!new_args[0]->IsInstance<TempExprNode>());
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
                           const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = Cast(n->data, cfg->dtype_input);
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExpr(ret, n->dom_scale, cfg->dtype_input);
  }
  CHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.max_pool2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", CastDtypeInputRealize);


Expr AvgPoolRealize(const Call& ref_call,
                    const Array<Expr>& new_args,
                    const ObjectRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = n->data;
    if (n->dtype != cfg->dtype_activation) {
      data = Cast(n->data, cfg->dtype_activation);
    }
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExpr(ret, n->dom_scale, cfg->dtype_activation);
  }
  CHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.avg_pool2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

RELAY_REGISTER_OP("nn.global_avg_pool2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

Expr CastHintRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const ObjectRef& ctx) {
  const auto param = ref_call->attrs.as<CastHintAttrs>();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = Cast(n->data, param->dtype);
    return QRealizeIntExpr(ret, n->dom_scale, param->dtype);
  }
  CHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("annotation.cast_hint")
.set_attr<FForwardRewrite>("FQRealizeRewrite", CastHintRealize);

Pass QuantizeRealizePass() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(
          ForwardRewrite(f, "FQRealizeRewrite", nullptr, nullptr));
  };
  return CreateFunctionPass(pass_func, 1, "QuantizeRealize", {});
}

TVM_REGISTER_GLOBAL("relay._quantize.QuantizeRealize")
.set_body_typed(QuantizeRealizePass);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
