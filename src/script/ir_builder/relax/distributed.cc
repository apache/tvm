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
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/struct_info.h>
#include <tvm/script/ir_builder/relax/ir.h>
#include <tvm/tir/op.h>

#include "./utils.h"

namespace tvm {
namespace relax {
Expr MakeCallTIRDist(Expr func, Tuple args, Array<distributed::DTensorStructInfo> out_sinfo_list,
                     Optional<Expr> packed_ints) {
  for (const distributed::DTensorStructInfo& sinfo : out_sinfo_list) {
    const auto* shape = sinfo->tensor_sinfo->shape.as<ShapeExprNode>();
    CHECK(shape != nullptr) << "out_sinfo of call_tir should have defined ShapeExpr as shape. "
                               "However, one given structure info is "
                            << sinfo;
  }

  StructInfo out_sinfo{nullptr};
  if (out_sinfo_list.size() == 1) {
    out_sinfo = out_sinfo_list[0];
  } else {
    out_sinfo = TupleStructInfo({out_sinfo_list.begin(), out_sinfo_list.end()});
  }

  static const Op& op = Op::Get("relax.call_tir");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, {}, {out_sinfo});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, {}, {out_sinfo});
  }
  return call;
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.distributed.call_tir_dist")
    .set_body_typed(MakeCallTIRDist);

}  // namespace relax
}  // namespace tvm
