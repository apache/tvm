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
 * \file fp8_legalize.cc
 * \brief legalize fp8 type by adding cast_to_fp32
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cmath>
#include <tuple>

namespace tvm {
namespace tir {

class FP8ComputeLegalizer : public StmtExprMutator {
 public:
  PrimFunc Legalize(PrimFunc func) {
    // TODO(zihao)
  }
 private:
  /*!
   * \brief promote float8 to fp32 and keep other values unchanged.
   * \note only e4m3 is supported for now.
   * \return The converted value.
   */
  PrimExpr PromoteFP8ToF32(PrimExpr value) {
    if (!value.dtype().is_float8()) return value;
    if (const CastNode* cast = value.as<CastNode>()) {
      if (cast->value.dtype() == DataType::Float(32)) return cast->value;
    }
    // TODO(zihao)
  }

  /*!
   * \brief Cast value from fp32 to fp8 and keep other values unchanged.
   * \param value The input value
   * \return The converted value.
   */
  
};

class FP8StorageLegalizer : public StmtExprMutator {
 public:
  PrimFunc Legalize(PrimFunc func) {
    // TODO(zihao)
  }
};

namespace transform {

Pass FP8ComputeLegalize() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return FP8ComputeLegalizer().Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FP8ComputeLegalize", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FP8ComputeLegalize").set_body_typed(FP8ComputeLegalize);

Pass FP8StorageLegalize() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return FP8StorageLegalizer().Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FP8StorageLegalize", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FP16StorageLegalize").set_body_typed(FP8StorageLegalize);

}  // namespace transform
}  // namespace tir
}  // namespace tvm