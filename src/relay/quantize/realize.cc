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

#include "./realize.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/transform.h>

#include "../op/annotation/annotation.h"
#include "../qnn/utils.h"
#include "./quantize.h"

#include "../transforms/pattern_utils.h"
#include "../qnn/op/op_common.h"
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>


namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

Expr QRealizeIntExprNode::Realize() const {
  printf("QRealizeIntExprNode\n");
  Expr data = this->data;
  // dequantize
  Expr zero_point = this->zero_point;
  Expr dom_scale = this->dom_scale;
  dom_scale = Cast(dom_scale, DataType::Float(32));

  data = Cast(data, DataType::Int(64));
  zero_point = Cast(zero_point, DataType::Int(64));
  data = Subtract(data, zero_point);

  data = Cast(data, DataType::Float(32));
  data = Multiply(data, dom_scale);
  printf("QRealizeIntExprNode done\n");
  return data;
}

QRealizeIntExpr::QRealizeIntExpr(Expr data, Expr dom_scale, Expr zero_point, DataType dtype) {
  ObjectPtr<QRealizeIntExprNode> n = make_object<QRealizeIntExprNode>();
  n->data = std::move(data);
  n->dom_scale = std::move(dom_scale);
  n->dtype = std::move(dtype);
  n->zero_point = std::move(zero_point);
  data_ = std::move(n);
}

inline Expr ForwardOp(const Call& ref_call, const Array<Expr>& args) {
  return Call(ref_call->op, args, ref_call->attrs, ref_call->type_args);
}

/* calculate `data * s1 / s2`, use shift if possible */
inline Expr MulAndDiv(Expr data, float s1, float s2, DataType dtype,
                      const Array<IndexExpr>& data_shape) {
  const QConfig& cfg = QConfig::Current();
  // here we assume the dtype of data is dtype activation
  if (s1 == s2) return data;

  float factor = s1 / s2;
  float shift_factor = std::log2(factor);
  ICHECK_GT(shift_factor, 0);
  if (static_cast<int>(shift_factor) == shift_factor) {
    return (LeftShift(data, MakeConstantScalar(dtype, static_cast<int>(shift_factor))));
  } else if (static_cast<int>(factor) == factor) {
    return (Multiply(data, MakeConstantScalar(dtype, factor)));
  } else {
    if (cfg->rounding == "UPWARD") {
      int32_t fixed_point_multiplier, shift;
      std::tie(fixed_point_multiplier, shift) = qnn::GetFixedPointMultiplierShift(factor);
      data = (relay::FixedPointMultiply(data, fixed_point_multiplier, shift));
    } else {
      data = (qnn::FixedPointMultiplyToNearest(data, factor, data_shape));
    }

    return Cast(data, dtype);
  }
}

inline Expr MulAndDiv_nobias(Expr data, float s1, float s2, DataType dtype,
                      const Array<IndexExpr>& data_shape) {
  const QConfig& cfg = QConfig::Current();
  // here we assume the dtype of data is dtype activation
  if (s1 == s2) return data;

  float factor = s1 / s2;
  float shift_factor = std::log2(factor);
  ICHECK_GT(shift_factor, 0);
  if (static_cast<int>(shift_factor) == shift_factor) {
    return LeftShift(data, MakeConstantScalar(dtype, static_cast<int>(shift_factor)));
  } else if (static_cast<int>(factor) == factor) {
    return Multiply(data, MakeConstantScalar(dtype, factor));
  } else {
    if (cfg->rounding == "UPWARD") {
      int32_t fixed_point_multiplier, shift;
      std::tie(fixed_point_multiplier, shift) = qnn::GetFixedPointMultiplierShift(factor);
      data = (relay::FixedPointMultiply(data, fixed_point_multiplier, shift));
    } else {
      data = (qnn::FixedPointMultiplyToNearest(data, factor, data_shape));
    }

    return Cast(data, dtype);
  }
}


Expr QuantizeRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("quantizerealize\n");
  const QConfig& cfg = QConfig::Current();
  // do not handle data type cast
  const auto param = ref_call->attrs.as<SimulatedQuantizeAttrs>();
  ICHECK_EQ(param->rounding, "round");

  Expr dom_scale = new_args[1];
  Expr clip_min = new_args[2];
  Expr clip_max = new_args[3];
  Expr zero_point = new_args[4];

  //std::vector<float> dom_scale_imm_vector;
//测试：当没有per layer的时候
  //float dom_scale_imm = GetScalarFromConstant<float>(dom_scale);
  auto dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(dom_scale);
  float dom_scale_imm = static_cast<float>(dom_scales[0]);
  //if(cfg->per_channel && param->per_channel){
  //  for (size_t i = 0; i < dom_scales.size(); i++) {
  //    float inter_result = 1 / static_cast<float>(dom_scales[i]);
  //    dom_scale_imm_vector.push_back(inter_result);
  //  }
  //}
  float clip_min_imm = GetScalarFromConstant<float>(clip_min);
  float clip_max_imm = GetScalarFromConstant<float>(clip_max);

  auto zero_points = tvm::relay::qnn::GetFloatVectorFromConstant(zero_point);
  float zero_point_imm = static_cast<float>(zero_points[0]);

  // x * idom_scale = y * odom_scale
  // => y = x * idom_scale / odom_scale
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    printf("int32->int8\n");
    // int32->int8
    Expr data = n->data;
    
    auto idom_scale_imms = tvm::relay::qnn::GetFloatVectorFromConstant(n->dom_scale);
    float idom_scale_imm = static_cast<float>(idom_scale_imms[0]);
    auto odom_scale_imms = tvm::relay::qnn::GetFloatVectorFromConstant(dom_scale);
    float odom_scale_imm = static_cast<float>(odom_scale_imms[0]);

    if (idom_scale_imm == odom_scale_imm) {
      printf("1\n");
      // same domain scale, only clip
      data = Add(data, (MakeConstantScalar(n->dtype, static_cast<int>(zero_point_imm))));
      data = Clip(data, clip_min_imm, clip_max_imm);
      data = Clip(data, clip_min_imm, clip_max_imm);
      printf("int32->int8 done\n");
      return QRealizeIntExpr(data, dom_scale, zero_point, n->dtype);
    }

    float shift_nbit = std::log2(odom_scale_imm / idom_scale_imm);
    ICHECK_NE(shift_nbit, 0);
    if (static_cast<int>(shift_nbit) == shift_nbit) {
      printf("2\n");
      if (shift_nbit > 0) {
        // use right shift
        if (cfg->round_for_shift) {
          float round_bias = std::pow(2.0, shift_nbit - 1);
          data = Add(data, MakeConstantScalar(cfg->dtype_activation, static_cast<int>(round_bias)));
        }
        data = RightShift(data,
                          MakeConstantScalar(cfg->dtype_activation, static_cast<int>(shift_nbit)));
      } else {
        data = LeftShift(data,
                         MakeConstantScalar(cfg->dtype_activation, static_cast<int>(-shift_nbit)));
      }
      data = Add(data, (MakeConstantScalar(n->dtype, static_cast<int>(zero_point_imm))));
      data = Clip(data, clip_min_imm, clip_max_imm);
      printf("int32->int8 done\n");
      return QRealizeIntExpr(data, dom_scale, zero_point, n->dtype);
    } else {
      data = Cast(data, DataType::Int(64));
      if(cfg->per_channel){
        printf("3\n");
        //data = Cast(data, DataType::Float(64));
        data = Cast(data, DataType::Float(32));
        //if 这个simulated_quantize算子！是！跟在conv后面
        // printf("\nsi/so:\n");
        // for(int i=0; i<idom_scale_imms.size(); i++){
        //   printf("%lf",idom_scale_imms[i]);
        // }
        data = Multiply(data, CheckPointSiso(Divide(Cast(n->dom_scale, DataType::Float(32)), dom_scale)));
        //data = Multiply(data, (Divide(CheckPoint(Cast(n->dom_scale, DataType::Float(32))), dom_scale)));
        data = Cast(Round(data), DataType::Int(64));
      }
      else{
        if (cfg->rounding == "UPWARD") {
          int32_t fixed_point_multiplier, shift;
          std::tie(fixed_point_multiplier, shift) =
              qnn::GetFixedPointMultiplierShift(idom_scale_imm / odom_scale_imm);
          data = relay::FixedPointMultiply(data, fixed_point_multiplier, shift);
        } else {
          data = qnn::FixedPointMultiplyToNearest(data, idom_scale_imm / odom_scale_imm,
                                                ref_call->type_as<TensorTypeNode>()->shape);
        }
      }
      data = Add(data, (MakeConstantScalar(DataType::Int(64), zero_point_imm)));
      data = (Cast(Clip(data, clip_min_imm, clip_max_imm), n->dtype));
      //data = Cast(Clip(data, clip_min_imm, clip_max_imm), n->dtype);
      printf("int32->int8 done\n");
      return QRealizeIntExpr(data, dom_scale, zero_point, n->dtype);
    }
  }

  // quantize from real
  printf("quantize from real\n");
  Expr round_data;
  Expr scaled_data;
  Expr zp_added;
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  Expr data = new_args[0];

  

  if(param->per_channel && ref_call->attrs.as<SimulatedQuantizeAttrs>()->kind == kQWeight ){
    //scaled_data = Multiply(data, MakeConstantTensor(DataType::Float(32), {(int64_t)dom_scales.size(), 1, 1, 1}, dom_scale_imm_vector));
    //zp_added = Add(scaled_data, MakeConstantTensor(DataType::Float(32), {(int64_t)zero_points.size()}, zero_points));
    scaled_data = Divide(data, Cast(dom_scale, DataType::Float(32) ));
    zp_added = Add(scaled_data, (zero_point));
  }
  else{
    scaled_data = Multiply(data, MakeConstantScalar(DataType::Float(32), 1 / dom_scale_imm));
    zp_added = Add(scaled_data, MakeConstantScalar(DataType::Float(32), zero_point_imm));
  }
  

  if(ref_call->attrs.as<SimulatedQuantizeAttrs>()->kind == kQInput){
    round_data = (Clip(Round(zp_added), clip_min_imm, clip_max_imm));
    //round_data = Clip(Round(zp_added), clip_min_imm, clip_max_imm);
    }

  else if (ref_call->attrs.as<SimulatedQuantizeAttrs>()->kind == kQBias){
    round_data = zp_added;}
  else{
    round_data = Clip(Round(zp_added), clip_min_imm, clip_max_imm);}
  
  printf("quantize from real done\n");
  return QRealizeIntExpr(round_data, dom_scale, zero_point, DataType::Float(32));
}

Expr FoldConstantOpt(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::FoldConstant()(mod);
  auto entry_func = Downcast<Function>(mod->Lookup("main"));
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

RELAY_REGISTER_OP("relay.op.annotation.simulated_quantize")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", QuantizeRealize);

Expr Conv2dRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("Conv2dRealize\n");
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;

    Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。

    auto zps_data = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->zero_point);
    float zps_data_imm = static_cast<float>(zps_data[0]);

    auto zps_weight = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->zero_point);
    float zps_weight_imm = static_cast<float>(zps_weight[0]);   

    auto lhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->dom_scale);
    float lhs_dom_scale_imm = static_cast<float>(lhs_dom_scales[0]);
    auto rhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->dom_scale);
    float rhs_dom_scale_imm = static_cast<float>(rhs_dom_scales[0]);

    if (lhs->dtype != cfg->dtype_input) {
      ldata = Cast(ldata, cfg->dtype_input);
    }
    Expr rdata = Cast(rhs->data, cfg->dtype_weight);

    ldata = Subtract(ldata, MakeConstantScalar(cfg->dtype_input, zps_data_imm));
    if(cfg->per_channel){
      rdata = Subtract(rdata, MakeConstantTensor(cfg->dtype_weight, {(int64_t)zps_weight.size(),1,1,1}, zps_weight));
      //rdata = Subtract(rdata, Cast(rhs->zero_point, cfg->dtype_weight));  //(192,64,5,5) -(192,1,1,1) 
    }
    else{
      rdata = Subtract(rdata, MakeConstantScalar(cfg->dtype_weight, zps_weight_imm)); ////
    }
    

    const auto ref_attrs = ref_call->attrs.as<Conv2DAttrs>();
    auto attrs = make_object<Conv2DAttrs>();
    *attrs = *ref_attrs;
    DataType out_dtype = cfg->dtype_activation;
    attrs->out_dtype = out_dtype;

    Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
    //当per_channel的时候，rhs的scale就是[, , , , .....]
    Expr mul;
    if(cfg->per_channel){
      mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), MakeConstantTensor(DataType::Float(32), {(int64_t)rhs_dom_scales.size(),1,1}, rhs_dom_scales));
      //mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), rhs->dom_scale);
    }
    else{
      mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), MakeConstantScalar(DataType::Float(32), rhs_dom_scale_imm));
    }////////
    Expr dom_scale = FoldConstantOpt(mul);
    printf("Conv2dRealize done\n");
    return QRealizeIntExpr(ret, dom_scale, zero_point, out_dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.conv2d").set_attr<FForwardRewrite>("FQRealizeRewrite", Conv2dRealize);

Expr Conv1dRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("Conv1dRealize\n");
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

  Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
  auto zps_data = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->zero_point);
  float zps_data_imm = static_cast<float>(zps_data[0]);

  auto zps_weight = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->zero_point);
  float zps_weight_imm = static_cast<float>(zps_weight[0]);


  auto lhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->dom_scale);
  float lhs_dom_scale_imm = static_cast<float>(lhs_dom_scales[0]);
  auto rhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->dom_scale);
  float rhs_dom_scale_imm = static_cast<float>(rhs_dom_scales[0]);


  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);


  ldata = Subtract(ldata, MakeConstantScalar(cfg->dtype_input, zps_data_imm));

  if(cfg->per_channel){
    //rdata = Subtract(rdata, MakeConstantTensor(cfg->dtype_weight, {(int64_t)zps_weight.size()}, zps_weight));
    rdata = Subtract(rdata, Cast(rhs->zero_point, cfg->dtype_weight));
  }
  else{
    rdata = Subtract(rdata, MakeConstantScalar(cfg->dtype_weight, zps_weight_imm));
  }
  const auto ref_attrs = ref_call->attrs.as<Conv1DAttrs>();
  auto attrs = make_object<Conv1DAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul;
  if(cfg->per_channel){
    //mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), MakeConstantTensor(DataType::Float(32), {(int64_t)rhs_dom_scales.size()}, rhs_dom_scales));
    mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), rhs->dom_scale);
  }
  else{
    mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), MakeConstantScalar(DataType::Float(32), rhs_dom_scale_imm));
  }
  Expr dom_scale = FoldConstantOpt(mul);
  printf("Conv1dRealize done\n");
  return QRealizeIntExpr(ret, dom_scale, zero_point, out_dtype);
}

RELAY_REGISTER_OP("nn.conv1d").set_attr<FForwardRewrite>("FQRealizeRewrite", Conv1dRealize);

Expr DenseRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("DenseRealize\n");
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();

  Expr ldata = lhs->data;

  Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一
  auto zps_data = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->zero_point);
  float zps_data_imm = static_cast<float>(zps_data[0]);

  auto zps_weight = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->zero_point);
  float zps_weight_imm = static_cast<float>(zps_weight[0]);

  auto lhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->dom_scale);
  float lhs_dom_scale_imm = static_cast<float>(lhs_dom_scales[0]);
  auto rhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->dom_scale);
  float rhs_dom_scale_imm = static_cast<float>(rhs_dom_scales[0]);

  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<DenseAttrs>();
  auto attrs = make_object<DenseAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;
  ldata = Subtract(ldata, MakeConstantScalar(cfg->dtype_input, zps_data_imm));
  rdata = Subtract(rdata, MakeConstantScalar(cfg->dtype_weight, zps_weight_imm));

  Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), MakeConstantScalar(DataType::Float(32), rhs_dom_scale_imm));
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExpr(ret, dom_scale, zero_point, out_dtype);
  printf("DenseRealize done\n");
}

RELAY_REGISTER_OP("nn.dense").set_attr<FForwardRewrite>("FQRealizeRewrite", DenseRealize);

Expr MulRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("MulRealize\n");
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    // execute the operation with activation data type.
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;
    Expr rdata = rhs->data;

    Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一
    auto zps_data = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->zero_point);
    float zps_data_imm = static_cast<float>(zps_data[0]);

    auto zps_weight = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->zero_point);
    float zps_weight_imm = static_cast<float>(zps_weight[0]);

    auto lhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->dom_scale);
    float lhs_dom_scale_imm = static_cast<float>(lhs_dom_scales[0]);
    auto rhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->dom_scale);
    float rhs_dom_scale_imm = static_cast<float>(rhs_dom_scales[0]);

    DataType dtype = cfg->dtype_activation;
    if (lhs->dtype != dtype) {
      ldata = Cast(ldata, dtype);
    }
    if (rhs->dtype != dtype) {
      rdata = Cast(rdata, dtype);
    }

    ldata = Subtract(ldata, MakeConstantScalar(dtype, zps_data_imm));
    rdata = Subtract(rdata, MakeConstantScalar(dtype, zps_weight_imm)); 
    Expr ret = ForwardOp(ref_call, {ldata, rdata});
    Expr mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), MakeConstantScalar(DataType::Float(32), rhs_dom_scale_imm));
    Expr dom_scale = FoldConstantOpt(mul);
    return QRealizeIntExpr(ret, dom_scale, zero_point, dtype);
    printf("MulRealize done\n");
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("multiply").set_attr<FForwardRewrite>("FQRealizeRewrite", MulRealize);

float ChooseDomScale(const Array<Expr>& ref_args, const std::vector<const QRealizeIntExprNode*>& nptrs) {
  printf("ChooseDomScale\n");
  if (nptrs.size() == 2) {
    // x = a * s1, y = b * s2
    // x + y = (a * s1 / s2 + b) * s2, if s1 > s2
    //       = (a + b * s2 / s1) * s1, if s2 > s1
    
    auto scale1 = tvm::relay::qnn::GetFloatVectorFromConstant(nptrs[0]->dom_scale);
    auto scale2 = tvm::relay::qnn::GetFloatVectorFromConstant(nptrs[1]->dom_scale);
    float s1 = static_cast<float>(scale1[0]);
    float s2 = static_cast<float>(scale2[0]);
    float s;
    
    s = s1 > s2 ? s2 : s1;
    return s;
    //float s1 = GetScalarFromConstant<float>(nptrs[0]->dom_scale);
    //float s2 = GetScalarFromConstant<float>(nptrs[1]->dom_scale);
    printf("ChooseDomScale done\n");
    
  } else {
    const QConfig& cfg = QConfig::Current();
    float scale = cfg->global_scale;
    printf("ChooseDomScale done\n");
    return scale / std::pow(2.0, cfg->nbit_activation - 1);
  }
}
template <typename ValueType = int64_t>
std::vector<ValueType> GetConcrete(const Array<PrimExpr>& vals) {
  std::vector<ValueType> concrete;
  for (const auto& v : vals) {
    auto* val = v.as<IntImmNode>();
    ICHECK(val);
    concrete.push_back(val->value);
  }
  return concrete;
}



/* \brief Unify the dom scale of arguments */
Array<Expr> UnifyDTypeScale(const Array<Expr>& ref_args, const Array<Expr>& args,
                            DataType* dtype_ptr, Expr* scale_ptr,
                            DataType dtype = DataType::Void()) {
  printf("UnifyDTypeScale\n");
  static const Op& simulated_quantize = Op::Get("relay.op.annotation.simulated_quantize");
  const QConfig& cfg = QConfig::Current();
  std::vector<const QRealizeIntExprNode*> nptrs;
  Array<Expr> ret;
  for (auto arg : args) {
    const auto* nptr = arg.as<QRealizeIntExprNode>();
    ICHECK(nptr);
    nptrs.push_back(nptr);
    ret.push_back(nptr->data);
  }
  Array<Expr> zp;
  std::vector<float> zps_imm;
  for (size_t i = 0; i < ret.size(); ++i) {
    auto zps_data = tvm::relay::qnn::GetFloatVectorFromConstant(args[i].as<QRealizeIntExprNode>()->zero_point);
    float zps_data_imm = static_cast<float>(zps_data[0]);
    zps_imm.push_back(zps_data_imm);
  }
  for (size_t i = 0; i < ret.size(); ++i) {
    zp.push_back(MakeConstantScalar(cfg->dtype_weight, zps_imm[i]));
  }

  // unify the data type
  ICHECK_EQ(ref_args.size(), args.size());

  if (dtype.is_void()) {
    if (ret.size() == 2 && nptrs[1]->dtype == cfg->dtype_input) {
      dtype = cfg->dtype_input;
    } else {
      dtype = cfg->dtype_activation;
    }
  }
  printf("add unify type\n");
  Op opp = Downcast<Op>(ref_args[1].as<CallNode>()->op);
  if(opp->name == "relay.op.annotation.simulated_quantize"){
    printf("add unify type into\n");
    zp.Set(0,Cast(zp[0],dtype));
    zp.Set(1,Cast(zp[1],DataType::Float(32)));
    ret.Set(0,Cast(ret[0],dtype));
    ret.Set(1,Cast(ret[1],DataType::Float(32)));
    printf("add unify type into done\n");
  }
  else{
    for (size_t i = 0; i < ret.size(); ++i) {
    auto ref_arg = ref_args[i].as<CallNode>();
    zp.Set(i,Cast(zp[i], dtype));
    if (nptrs[i]->dtype != dtype) {
      ret.Set(i, Cast(ret[i], dtype));
    } else if (ref_arg && ref_arg->op.same_as(simulated_quantize) &&
               ref_arg->attrs.as<SimulatedQuantizeAttrs>()->kind == kQInput) {
      auto new_arg = Cast(ret[i], cfg->dtype_input);
      new_arg = StopFusion(new_arg);
      ret.Set(i, Cast(new_arg, dtype));
    }
  }
  }

  float s;
  std::vector<float> s_vector;
  Expr dom_scale;
  // unify the dom_scale
  printf("###1");
  printf("###2");
  //printf( "add[0].dom_scale->shape.size()=%d\n",static_cast<int>(nptrs[0]->dom_scale->type_as<TensorTypeNode>()->shape.size()) );
  Op op = Downcast<Op>(ref_args[1].as<CallNode>()->op);
  //printf("%s",op->name.c_str);
  if(op->name == "relay.op.annotation.simulated_quantize"){
    if(ref_args[1].as<CallNode>()->attrs.as<SimulatedQuantizeAttrs>()->kind == kQBias && cfg->per_channel && nptrs[0]->dom_scale->type_as<TensorTypeNode>()->shape.size()!=0 ){
      printf("add in bias with per_channels\n");
      s_vector = tvm::relay::qnn::GetFloatVectorFromConstant(nptrs[0]->dom_scale);
      std::vector<float> output_multipliers;
      for (size_t i = 0; i < s_vector.size(); i++) {
        float multiplier = 1 / static_cast<float>(s_vector[i]);
        output_multipliers.push_back(multiplier);
      }
      //Expr output_multiplier = MakeConstantTensor(DataType::Float(32), {(int64_t)s_vector.size(),1,1}, output_multipliers);
      //Expr output_multiplier = MakeConstantTensor(DataType::Float(32), {(int64_t)tvm::relay::qnn::GetFloatVectorFromConstant(ret[1])->shape}, output_multipliers);
      Expr output_multiplier = MakeConstantTensor(DataType::Float(32), GetConcrete(ref_args[1]->type_as<TensorTypeNode>()->shape), output_multipliers);
      dom_scale = nptrs[0]->dom_scale;
      ret.Set(0,Subtract(ret[0], zp[0]));
      ret.Set(1,(Subtract(ret[1], zp[1])));
      //ret.Set(1,Divide(Cast(ret[1],DataType::Float(32)), Cast(dom_scale, DataType::Float(32))));
      ret.Set(1,CheckPointBiasS(Multiply(Cast(ret[1],DataType::Float(32)), output_multiplier)));// add[1] = bias/dom_scale = (192,1,1)*(192,1,1)
      ret.Set(1, Cast(Round(ret[1]), dtype));
      printf("add in bias with per_channels done\n");
    }
    //当遇到shi
    else if(ref_args[1].as<CallNode>()->attrs.as<SimulatedQuantizeAttrs>()->kind == kQBias && (!cfg->per_channel || nptrs[0]->dom_scale->type_as<TensorTypeNode>()->shape.size()==0 )){
      printf("add in bias without per_channel\n");
      s_vector = tvm::relay::qnn::GetFloatVectorFromConstant(nptrs[0]->dom_scale);
      s = static_cast<float>(s_vector[0]);
      //dom_scale = MakeConstantScalar(dtype, s);
      dom_scale = MakeConstantScalar(DataType::Float(32), s);
      ret.Set(0,Subtract(ret[0], zp[0]));
      ret.Set(1,(Subtract(ret[1], zp[1])));   
      //ret.Set(1,Divide(ret[1], dom_scale));
      ret.Set(1,CheckPointBiasS(Divide(Cast(ret[1], DataType::Float(32)), dom_scale )));
      ret.Set(1, Cast(Round(ret[1]), dtype));
      printf("add in bias without per_channel done\n");
    }
    else{
      printf("add fixedpoint mul");
      s = ChooseDomScale(ref_args,nptrs);
      dom_scale = MakeConstantScalar(DataType::Float(32), s);
      for (size_t i = 0; i < ret.size(); ++i) {
        auto cur_ss = tvm::relay::qnn::GetFloatVectorFromConstant(nptrs[i]->dom_scale);
        float cur_s = static_cast<float>(cur_ss[0]);
        //float cur_s = GetScalarFromConstant<float>(nptrs[i]->dom_scale);
        ret.Set(i,Subtract(ret[i], zp[i]));
        ret.Set(i, MulAndDiv(ret[i], cur_s, s, dtype, ref_args[i]->type_as<TensorTypeNode>()->shape));
      }
    }
  }
  else{
    printf("last is not simulated_quantize");
    s = ChooseDomScale(ref_args,nptrs);
    dom_scale = MakeConstantScalar(DataType::Float(32), s);
    for (size_t i = 0; i < ret.size(); ++i) {
      auto cur_ss = tvm::relay::qnn::GetFloatVectorFromConstant(nptrs[i]->dom_scale);
      float cur_s = static_cast<float>(cur_ss[0]);
      //float cur_s = GetScalarFromConstant<float>(nptrs[i]->dom_scale);
      ret.Set(i,Subtract(ret[i], zp[i]));
      ret.Set(i, MulAndDiv_nobias(ret[i], cur_s, s, dtype, ref_args[i]->type_as<TensorTypeNode>()->shape));
    }
  }
  *dtype_ptr = dtype;
  *scale_ptr = dom_scale;
  printf("UnifyDTypeScale done\n");
  return ret;
}

Expr AddRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("AddRealize\n");
  ICHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    // execute the operation with activation data type.
    const QConfig& cfg = QConfig::Current();
    Array<Expr> ret_args =
        UnifyDTypeScale(ref_call->args, new_args, &dtype, &dom_scale, cfg->dtype_activation);
    for (size_t i = 0; i < ret_args.size(); ++i) {
      // do not fuse float32 arg
      if (new_args[i].as<QRealizeIntExprNode>()->dtype == DataType::Float(32)) {
        ret_args.Set(i, StopFusion(ret_args[i]));
      }
    }
    Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
    Expr ret;
    ret = ForwardOp(ref_call, ret_args);
    
    printf("AddRealize done\n");
    return QRealizeIntExpr(ret, dom_scale, zero_point, dtype);
  }

  ICHECK(!new_args[0]->IsInstance<TempExprNode>() && !new_args[1]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("add").set_attr<FForwardRewrite>("FQRealizeRewrite", AddRealize);

Expr ClipRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("ClipRealize\n");
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {

    auto zero_points = tvm::relay::qnn::GetFloatVectorFromConstant(n->zero_point);
    double zero_point_imm = static_cast<double>(zero_points[0]);

    const auto ref_attrs = ref_call->attrs.as<ClipAttrs>();
    auto attrs = make_object<ClipAttrs>();
    //double dom_scale = GetScalarFromConstant<float>(n->dom_scale);
    auto dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(n->dom_scale);
    double dom_scale = static_cast<float>(dom_scales[0]);
    attrs->a_min = ref_attrs->a_min / dom_scale - zero_point_imm;
    attrs->a_max = ref_attrs->a_max / dom_scale - zero_point_imm;

    Expr ret = Call(ref_call->op, {n->data}, Attrs(attrs), ref_call->type_args);
    Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
    printf("ClipRealize done\n");
    return QRealizeIntExpr(ret, n->dom_scale, zero_point, n->dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("clip").set_attr<FForwardRewrite>("FQRealizeRewrite", ClipRealize);

Expr ConcatenateRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("ConcatenateRealize\n");
  ICHECK_EQ(new_args.size(), 1);
  ICHECK_EQ(ref_call->args.size(), 1);

  const auto* tuple = new_args[0].as<TupleNode>();
  const auto* ref_tuple = ref_call->args[0].as<TupleNode>();
  ICHECK(tuple);
  ICHECK(ref_tuple);
  const Array<Expr>& arr = tuple->fields;
  const Array<Expr>& ref_arr = ref_tuple->fields;

  if (arr[0].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    Array<Expr> ret_args = UnifyDTypeScale(ref_arr, arr, &dtype, &dom_scale);
    Expr ret = ForwardOp(ref_call, {Tuple(ret_args)});
    printf("ConcatenateRealize done\n");
    Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
    return QRealizeIntExpr(ret, dom_scale, zero_point, dtype);
  } else {
    for (auto arg : new_args) {
      ICHECK(!arg->IsInstance<TempExprNode>());
    }
    return Expr(nullptr);
  }
}

RELAY_REGISTER_OP("concatenate").set_attr<FForwardRewrite>("FQRealizeRewrite", ConcatenateRealize);

/* \brief forward the original operator */
Expr IdentityRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("IdentityRealize\n");
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = ForwardOp(ref_call, {n->data});
    //Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
    Expr zero_point = n -> zero_point;
    printf("IdentityRealize done\n");
    return QRealizeIntExpr(ret, n->dom_scale, zero_point, n->dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.relu").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("reshape").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("strided_slice").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("nn.batch_flatten")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("transpose").set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("annotation.stop_fusion")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

/* \brief for unary operators which requantize its input to dtype_nbit */
Expr CastDtypeInputRealize(const Call& ref_call, const Array<Expr>& new_args,
                           const ObjectRef& ctx) {
  printf("CastDtypeInputRealize\n");
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = Cast(n->data, cfg->dtype_input);

    auto zero_points = tvm::relay::qnn::GetFloatVectorFromConstant(n->zero_point);
    float zero_point_imm = static_cast<float>(zero_points[0]);
    Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
    data = Subtract(data, MakeConstantScalar(cfg->dtype_input, zero_point_imm));
    Expr ret = ForwardOp(ref_call, {data});
    printf("CastDtypeInputRealize done\n");
    return QRealizeIntExpr(ret, n->dom_scale, zero_point, cfg->dtype_input);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.max_pool2d")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", CastDtypeInputRealize);

RELAY_REGISTER_OP("nn.max_pool1d")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", CastDtypeInputRealize);

Expr AvgPoolRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("AvgPoolRealize\n");
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = n->data;
    Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
    if (n->dtype != cfg->dtype_activation) {
      data = Cast(n->data, cfg->dtype_activation);
    }
    auto zero_points = tvm::relay::qnn::GetFloatVectorFromConstant(n->zero_point);
    float zero_point_imm = static_cast<float>(zero_points[0]);
    data = Subtract(data, MakeConstantScalar(cfg->dtype_activation, zero_point_imm));

    Expr ret = (ForwardOp(ref_call, {data}));
    printf("AvgPoolRealize done\n");
    return QRealizeIntExpr(ret, n->dom_scale, zero_point, cfg->dtype_activation);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.avg_pool2d").set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

RELAY_REGISTER_OP("nn.global_avg_pool2d")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

Expr CastHintRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  printf("CastHintRealize\n");
  const auto param = ref_call->attrs.as<CastHintAttrs>();
  ICHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = Cast(n->data, param->dtype);
    Expr zero_point = n->zero_point;
    //Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
    printf("CastHintRealize done\n");
    return QRealizeIntExpr(ret, n->dom_scale, zero_point, param->dtype);
  }
  ICHECK(!new_args[0]->IsInstance<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("annotation.cast_hint")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", CastHintRealize);

Expr BatchMatmulRealize(const Call& ref_call, const Array<Expr>& new_args, const ObjectRef& ctx) {
  //printf("BatchMatmulRealize\n");
  const QConfig& cfg = QConfig::Current();
  ICHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->IsInstance<TempExprNode>() || !new_args[1]->IsInstance<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();

  Expr ldata = lhs->data;
  Expr rdata = rhs->data;
  DataType dtype_input = cfg->dtype_input;
  DataType dtype_weight = cfg->dtype_weight;

  auto zps_data = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->zero_point);
  float zps_data_imm = static_cast<float>(zps_data[0]);

  auto zps_weight = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->zero_point);
  float zps_weight_imm = static_cast<float>(zps_weight[0]);


  auto lhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(lhs->dom_scale);
  float lhs_dom_scale_imm = static_cast<float>(lhs_dom_scales[0]);
  auto rhs_dom_scales = tvm::relay::qnn::GetFloatVectorFromConstant(rhs->dom_scale);
  float rhs_dom_scale_imm = static_cast<float>(rhs_dom_scales[0]);

  if (lhs->dtype != dtype_input) {
    ldata = Cast(ldata, dtype_input);
  }
  if (rhs->dtype != dtype_weight) {
    rdata = Cast(rdata, dtype_weight);
  }

  const auto ref_attrs = ref_call->attrs.as<BatchMatmulAttrs>();
  auto attrs = make_object<BatchMatmulAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;
  ldata = Subtract(ldata, MakeConstantScalar(cfg->dtype_input, zps_data_imm));
  rdata = Subtract(rdata, MakeConstantScalar(cfg->dtype_weight, zps_weight_imm));

  Expr ret = Call(ref_call->op, {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr mul = Multiply(MakeConstantScalar(DataType::Float(32), lhs_dom_scale_imm), MakeConstantScalar(DataType::Float(32), rhs_dom_scale_imm));
  Expr dom_scale = FoldConstantOpt(mul);
  Expr zero_point = MakeConstantScalar(DataType::Float(32), 0);//其实没必要存在，为了保证输入参数的统一。
  return QRealizeIntExpr(ret, dom_scale, zero_point, out_dtype);
}

RELAY_REGISTER_OP("nn.batch_matmul")
    .set_attr<FForwardRewrite>("FQRealizeRewrite", BatchMatmulRealize);

Pass QuantizeRealizePass() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ForwardRewrite(f, "FQRealizeRewrite", nullptr, nullptr));
      };
  return CreateFunctionPass(pass_func, 1, "QuantizeRealize", {});
}

TVM_REGISTER_GLOBAL("relay._quantize.QuantizeRealize").set_body_typed(QuantizeRealizePass);
  
}  // namespace quantize
}  // namespace relay
}  // namespace tvm
