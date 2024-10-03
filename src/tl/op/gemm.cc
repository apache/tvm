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
 * \file tl/op/gemm.cc
 *
 * Define gemm operator.
 */

#include "gemm.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

static std::vector<int> toPrimeFactors(int x) {
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

Gemm::Gemm(Array<PrimExpr> args, BufferMap vmap) {
  call_args = args;
  A = vmap[GetVarFromAccessPtr(args[0])];
  B = vmap[GetVarFromAccessPtr(args[1])];
  C = vmap[GetVarFromAccessPtr(args[2])];
  trans_A = args[3].as<Bool>().value();
  trans_B = args[4].as<Bool>().value();
  M = args[5].as<IntImm>().value()->value;
  N = args[6].as<IntImm>().value()->value;
  K = args[7].as<IntImm>().value()->value;
  policy = static_cast<GemmWarpPolicy>(args[8].as<IntImm>().value()->value);
}

std::pair<int, int> Gemm::ComputeWarpPartition(int num_warps, Target target) const {
  int m_warp = 1, n_warp = 1;
  if (TargetIsHopper(target)) {
    ICHECK(num_warps % 4 == 0) << "Use Warp Group MMA requires 128*N threads.";
    if (this->policy == GemmWarpPolicy::kFullRow || this->policy == GemmWarpPolicy::kSquare) {
      m_warp = num_warps;
      ICHECK(this->M % num_warps == 0);
    } else if (this->policy == GemmWarpPolicy::kFullCol) {
      m_warp = 4;
      n_warp = num_warps / 4;
      ICHECK(this->N % n_warp == 0);
    } else {
      ICHECK(0) << "Unknown GemmWarpPolicy";
    }
    return {m_warp, n_warp};
  }
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

Stmt Gemm::Lower(const LowerArgs& T, arith::Analyzer* analyzer) const {
  ICHECK(T.block_size % 32 == 0);
  auto [warp_m, warp_n] = ComputeWarpPartition(T.block_size / 32, T.target);
  std::stringstream ss;
  std::string op_name = "tl::gemm_ss";
  if (A.scope() == "local.fragment") {
    ICHECK(B.scope() != "local.fragment");
    op_name = "tl::gemm_rs";
  } else if (B.scope() == "local.fragment") {
    op_name = "tl::gemm_sr";
  }
  ss << op_name << "<" << M << ", " << N << ", " << K << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << trans_A << ", " << trans_B << ">";

  auto A_buffer = T.buffer_remap.count(A) ? T.buffer_remap[A] : A;
  auto B_buffer = T.buffer_remap.count(B) ? T.buffer_remap[B] : B;
  auto C_buffer = T.buffer_remap[C];

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(A_buffer.access_ptr(1));
  new_args.push_back(B_buffer.access_ptr(1));
  new_args.push_back(C_buffer.access_ptr(3));
  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap Gemm::InferLayout(const LayoutInferArgs& T, InferLevel level) {
  if (completed_) return {};

  LayoutMap results;
  ICHECK(C.scope() == "local.fragment");
  auto [warp_m, warp_n] = ComputeWarpPartition(T.block_size / 32, T.target);

  if (TargetIsVolta(T.target)) {
    auto fragment = makeGemmVoltaFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment);
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      results.Set(A, makeGemmVoltaABLayout(*as_const_int(A->shape[0]), *as_const_int(A->shape[1]),
                                           true, trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      ICHECK(trans_A == false);
      results.Set(A, makeGemmVoltaFragmentA(M, N, K, M / warp_m, N / warp_n));
    } else {
      ICHECK(0);
    }

    ICHECK(B.scope() == "shared" || B.scope() == "shared.dyn");
    results.Set(B, makeGemmVoltaABLayout(*as_const_int(B->shape[0]), *as_const_int(B->shape[1]),
                                         false, trans_B ? 2 : 1));
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target)) {
    auto fragment = makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment);

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      results.Set(A, makeGemmABLayout(*as_const_int(A->shape[0]), *as_const_int(A->shape[1]),
                                      A->dtype.bits(), trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      ICHECK(trans_A == false);
      results.Set(A, makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n));
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      results.Set(B, makeGemmABLayout(*as_const_int(B->shape[0]), *as_const_int(B->shape[1]),
                                      B->dtype.bits(), trans_B ? 2 : 1));
    } else if (B.scope() == "local.fragment") {
      ICHECK(trans_B == false);
      results.Set(B, makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsHopper(T.target)) {
    auto fragment = makeGemmFragmentCHopper(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment);
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      results.Set(A, makeGemmABLayout(*as_const_int(A->shape[0]), *as_const_int(A->shape[1]),
                                      A->dtype.bits(), trans_A ? 1 : 2));
    } else {
      ICHECK(trans_A == false);
      results.Set(A, makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n));
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      results.Set(B, makeGemmABLayout(*as_const_int(B->shape[0]), *as_const_int(B->shape[1]),
                                      B->dtype.bits(), trans_B ? 2 : 1));
    } else {
      ICHECK(0) << "WGMMA only support B in shared.";
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str();
  }
  completed_ = true;
  return results;
}

TIR_REGISTER_TL_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace tl
}  // namespace tvm