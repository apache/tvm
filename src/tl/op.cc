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
 * \file tl/op.cc
 *
 * Define operators usd in tile library.
 */

#include "op.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "helper.h"

namespace tvm {
namespace tl {

using namespace tir;

#define TVM_TL_REGISTER_OP(OpName) \
  TVM_REGISTER_OP("tl." OpName).set_attr<TScriptPrinterName>("TScriptPrinterName", OpName)

#define TIR_DEFINE_TL_FUNC(OpName)                \
  const Op& OpName() {                            \
    static const Op& op = Op::Get("tl." #OpName); \
    return op;                                    \
  }                                               \
  TVM_TL_REGISTER_OP(#OpName)

TIR_DEFINE_TL_FUNC(gemm).set_num_inputs(5).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_FUNC(copy).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_FUNC(fill).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_FUNC(reduce).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_FUNC(region).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

static Var GetVarFromAccessPtr(const PrimExpr& expr) {
  auto call = expr.as<CallNode>();
  ICHECK(call);
  ICHECK(call->op.same_as(builtin::tvm_access_ptr()));
  auto var = call->args[1].as<VarNode>();
  ICHECK(var);
  return GetRef<Var>(var);
}

static Buffer GetBufferFromRegion(const PrimExpr& expr) {
  auto call = expr.as<CallNode>();
  ICHECK(call);
  ICHECK(call->op.same_as(region()));
  auto load = call->args[0].as<BufferLoadNode>();
  ICHECK(load);
  return load->buffer;
}

Array<Range> ParseRegionArgs(const CallNode* call) {
  Array<Range> results;
  ICHECK(call->op.same_as(region()));
  size_t n = call->args.size();
  size_t ndim = n - 2;
  auto load = call->args[0].as<BufferLoadNode>();
  ICHECK(load);
  ICHECK(load->indices.size() == ndim);
  for (size_t i = 0; i < ndim; i++) {
    PrimExpr min = load->indices[i];
    PrimExpr extent = call->args[2 + i];
    results.push_back(Range::FromMinExtent(min, extent));
  }
  return results;
}

GemmArgs GemmArgs::Parse(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap) {
  GemmArgs gemm_args;
  gemm_args.A = vmap[GetVarFromAccessPtr(args[0])];
  gemm_args.B = vmap[GetVarFromAccessPtr(args[1])];
  gemm_args.C = vmap[GetVarFromAccessPtr(args[2])];
  gemm_args.trans_A = args[3].as<Bool>().value();
  gemm_args.trans_B = args[4].as<Bool>().value();
  gemm_args.M = args[5].as<IntImm>().value()->value;
  gemm_args.N = args[6].as<IntImm>().value()->value;
  gemm_args.K = args[7].as<IntImm>().value()->value;
  gemm_args.policy = static_cast<GemmWarpPolicy>(args[8].as<IntImm>().value()->value);
  return gemm_args;
}

std::vector<int> toPrimeFactors(int x) {
  int i = 2;
  std::vector<int> result;
  while (x > 1) {
    if (x % i == 0) {
      x /= i;
      result.push_back(i);
    } else {
      i++;
    }
  }
  return result;
}

std::pair<int, int> GemmArgs::ComputeWarpPartition(int num_warps) const {
  int m_warp = 1, n_warp = 1;
  if (this->policy == GemmWarpPolicy::kFullRow) {
    m_warp = num_warps;
    ICHECK(this->M % num_warps == 0);
  } else if (this->policy == GemmWarpPolicy::kFullCol) {
    n_warp = num_warps;
    ICHECK(this->N % num_warps == 0);
  } else if (this->policy == GemmWarpPolicy::kSquare) {
    auto factors = toPrimeFactors(num_warps);
    for (int factor : factors) {
      bool M_divisible = (this->M % (factor * m_warp)) == 0;
      bool N_divisible = (this->N % (factor * n_warp)) == 0;
      if (M_divisible && N_divisible) {
        if (this->M / m_warp >= this->N / n_warp)
          m_warp *= factor;
        else
          n_warp *= factor;
      } else if (M_divisible) {
        m_warp *= factor;
      } else if (N_divisible) {
        n_warp *= factor;
      } else {
        ICHECK(0) << "Cannot compute warp partition for shape" << M << " " << N
                  << " with num_warps " << num_warps;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }
  // TODO: perform more checks here

  return {m_warp, n_warp};
}

CopyArgs CopyArgs::Parse(const Array<PrimExpr>& args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    rgs[i] = ParseRegionArgs(call);
    bf[i] = GetBufferFromRegion(expr);
  }
  CopyArgs copy_args;
  std::tie(copy_args.src, copy_args.dst) = std::tie(bf[0], bf[1]);
  std::tie(copy_args.src_range, copy_args.dst_range) = std::tie(rgs[0], rgs[1]);
  // check range equal
  copy_args.CheckRangeEqual();
  return copy_args;
}

bool CopyArgs::CheckRangeEqual() const {
  Array<Range> lhs, rhs;
  for (const auto& rg : src_range)
    if (!is_one(rg->extent)) lhs.push_back(rg);
  for (const auto& rg : dst_range)
    if (!is_one(rg->extent)) rhs.push_back(rg);
  return StructuralEqual()(lhs, rhs);
}

Array<IterVar> CopyArgs::MakeIterVars() const {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < src_range.size(); i++) {
    if (is_one(src_range[i]->extent)) continue;
    Var var = Var(std::string{char('i' + idx)});
    idx++;
    loop_vars.push_back({Range(0, src_range[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

Array<PrimExpr> CopyArgs::MakeIndices(const Array<IterVar>& ivs, int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  ICHECK(idx == ivs.size());
  return indices;
}

PrimExpr CopyArgs::MakePredicate(arith::Analyzer* analyzer, const Array<IterVar>& ivs,
                                 Array<PrimExpr> extents, int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent)) continue;
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++) cond = And(cond, cond_list[i]);
    return cond;
  }
}

FillArgs FillArgs::Parse(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap) {
  FillArgs fill_args;
  fill_args.dst = vmap[GetVarFromAccessPtr(args[0])];
  if (args[1]->dtype != fill_args.dst->dtype) {
    fill_args.value = Cast(fill_args.dst->dtype, args[1]);
  } else {
    fill_args.value = args[1];
  }
  return fill_args;
}

ReduceArgs ReduceArgs::Parse(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap) {
  ReduceArgs reduce_args;
  reduce_args.src = vmap[GetVarFromAccessPtr(args[0])];
  reduce_args.dst = vmap[GetVarFromAccessPtr(args[1])];
  String reduce_type = args[2].as<StringImm>().value()->value;
  reduce_args.dim = args[3].as<IntImm>().value()->value;
  if (reduce_type == "sum")
    reduce_args.type = ReduceType::kSum;
  else if (reduce_type == "max")
    reduce_args.type = ReduceType::kMax;
  else if (reduce_type == "min")
    reduce_args.type = ReduceType::kMin;
  else
    ICHECK(0) << "Unknown reduce type: " << reduce_type;
  reduce_args.clear = args[4].as<Bool>().value();
  return reduce_args;
}

PrimExpr ReduceArgs::MakeInitValue() const {
  switch (type) {
    case ReduceType::kSum:
      return make_zero(dst->dtype);
    case ReduceType::kMax:
      return make_const(dst->dtype, -INFINITY);
    case ReduceType::kMin:
      return make_const(dst->dtype, INFINITY);
    default:
      ICHECK(0);
  }
}

PrimExpr ReduceArgs::MakeReduce(const PrimExpr& a, const PrimExpr& b) const {
  PrimExpr lhs = a, rhs = b;
  if (lhs->dtype != rhs->dtype) {
    rhs = Cast(lhs->dtype, rhs);
  }
  switch (type) {
    case ReduceType::kSum:
      return lhs + rhs;
    case ReduceType::kMax:
      return Max(lhs, rhs);
    case ReduceType::kMin:
      return Min(lhs, rhs);
    default:
      ICHECK(0);
      return PrimExpr(0);
  }
}

std::string ReduceArgs::MakeCodegenReducer() const {
  switch (type) {
    case ReduceType::kSum:
      return "tl::SumOp";
    case ReduceType::kMax:
      return "tl::MaxOp";
    case ReduceType::kMin:
      return "tl::MinOp";
    default:
      ICHECK(0);
      return "";
  }
}

}  // namespace tl
}  // namespace tvm
