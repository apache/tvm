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
 * \file intrin_rule_cuda.cc
 * \brief CUDA intrinsic rules.
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, CUDA fast math.
using tir::FLowerIntrinsic;

struct CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
        case 64:
          return name;
        case 32:
          return name + 'f';
        case 16: {
          if (name == "fabs") {
            return "__habs";
          } else if (name == "round") {
            return "hrint";
          } else {
            return "h" + name;
          }
        }
        default:
          return "";
      }
    } else if (t.is_bfloat16()) {
      return 'h' + name;
    }
    return "";
  }
};

struct CUDAFastMath : public CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return CUDAMath::operator()(t, name);
    }
    return "";
  }
};

struct CUDAFastMathTan : public CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
        case 64:
          return name;
        // `__tanf` seems to produce some values too deviant from numpy tan version.
        // So, let's use just `tanf` instead.
        case 32:
          return name + 'f';
        case 16:
          return 'h' + name;
        default:
          return "";
      }
    }
    return "";
  }
};

struct CUDAPopcount {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_uint()) {
      switch (t.bits()) {
        case 32:
          return "__popc";
        case 64:
          return "__popcll";
        default:
          return "";
      }
    }
    return "";
  }
};

struct CUDAWarpIntrinsic {
  const Op operator()(DataType t, const Op& orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tir.cuda.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tir.cuda.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tir.cuda.__shfl_down_sync");
    }
  }
};

static PrimExpr DispatchCUDAWarpActiveMask(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  return Call(call->dtype, Op::Get("tir.cuda.__activemask"), call->args);
}

template <typename T>
static PrimExpr DispatchCUDAShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  Array<PrimExpr> cuda_args{{call->args[0], call->args[1], call->args[2], call->args[3]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), cuda_args);
}

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                                     DispatchPureExtern<CUDAFastMath>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.exp10")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>);

TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                                     DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                                     DispatchPureExtern<CUDAFastMath>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>);

TVM_REGISTER_OP("tir.tan").set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                                     DispatchPureExtern<CUDAFastMathTan>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                                     DispatchPureExtern<CUDAFastMath>);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                                     DispatchPureExtern<CUDAFastMath>);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.atan")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                                     DispatchPureExtern<CUDAMath>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAPopcount>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAShuffle<CUDAWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAShuffle<CUDAWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAShuffle<CUDAWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_activemask")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAWarpActiveMask);

TVM_REGISTER_OP("tir.fmod")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

// Register low-level builtin ops.
// TODO(tvm-team): consider make CUDA its own subfolder and create a file for low-level builtins.
TVM_REGISTER_OP("tir.cuda.__shfl_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane", "Expr", "The source thread id.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("cuda.need_warp_shuffle", true);

TVM_REGISTER_OP("tir.cuda.__shfl_up_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be added.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_up_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("cuda.need_warp_shuffle", true);

TVM_REGISTER_OP("tir.cuda.__shfl_down_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be subtracted.")
    .add_argument("width", "Expr", "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_down_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("cuda.need_warp_shuffle", true);

TVM_REGISTER_OP("tir.cuda.__activemask")
    .set_num_inputs(0)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__activemask")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<bool>("cuda.need_warp_shuffle", true);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
