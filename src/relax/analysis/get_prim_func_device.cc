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
 * \file get_prim_func_device.cc
 *
 * \brief Analysis to retrieve the device type of primfuncs.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

class PrimFuncGatherer : public ExprVisitor {
 public:
  explicit PrimFuncGatherer(const IRModule& m) : mod_(m) {}

  Map<GlobalVar, Integer> Track(const IRModule& mod) {
    PrimFuncGatherer gatherer(mod);
    // Go through each Relax function in the module.
    for (const auto& [gv, fn] : mod->functions) {
      if (const auto* func = fn.as<FunctionNode>()) {
        gatherer(GetRef<Function>(func));
      }
    }
    return gatherer.dev_gvar_map_;
  }

  void VisitExpr_(const CallNode* call) override {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (!call->op.same_as(call_tir_op)) {
      return;
    }
    const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
    auto args = Downcast<Tuple>(call->args[1])->fields;
    int device_type = -1;
    for (const auto& arg : args) {
      auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(arg);
      if (tinfo != nullptr && tinfo->vdevice.defined()) {
        auto vdevice = tinfo->vdevice.value();
        device_type = vdevice->target->GetTargetDeviceType();
        break;
      }
    }

    this->dev_gvar_map_.Set(global_var, device_type);
  }

 private:
  Map<GlobalVar, Integer> dev_gvar_map_;
  const IRModule& mod_;
};

Map<GlobalVar, Integer> GetPrimFuncDevice(const IRModule& m) {
  return PrimFuncGatherer(m).Track(m);
}

TVM_REGISTER_GLOBAL("relax.analysis.get_prim_func_device").set_body_typed(GetPrimFuncDevice);

}  // namespace relax
}  // namespace tvm
