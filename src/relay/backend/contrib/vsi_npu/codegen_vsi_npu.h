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
#ifndef TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_CODEGEN_VSI_NPU_H_
#define TVM_RELAY_BACKEND_CONTRIB_VSI_NPU_CODEGEN_VSI_NPU_H_

#include <tim/vx/context.h>
#include <tim/vx/graph.h>
#include <tim/vx/operation.h>
#include <tvm/ir/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>

#include "op_map/op_setup.h"

using namespace tvm::runtime;
using namespace tvm::relay::contrib::vsi_npu::op_map;

namespace tvm {
namespace relay {
namespace contrib {
namespace vsi_npu {

class VsiError {
  // TODO
};

struct RawGraphDef {
  std::shared_ptr<char> compiled_graph;
  uint32_t compiled_graph_size;
  std::vector<tim::vx::TensorSpec> inputs_spec;
  std::vector<tim::vx::TensorSpec> outputs_spec;
  std::vector<uint32_t> output_map;
};

class TensorMakerImpl : private ExprVisitor {
 public:
  //TensorMakerImpl(const IRModule& module, const GlobalVar& var) : VsiErrorReporter(module, var) {}
  TensorMakerImpl(const IRModule& module, const GlobalVar& var) : module_(module), var_(var) {}
  std::map<Expr, std::shared_ptr<OpSetup>> Create(const Expr& expr);

 private:
  void InferCall(const CallNode* cn);
  void VisitInferred(const Expr& expr);

  void VisitExpr_(const CallNode* cn) final;

  // TODO:
  void VisitExpr_(const TupleNode* tn) final;
  void VisitExpr_(const TupleGetItemNode* tg) final {
    LOG(INFO) << __FUNCTION__<< "TupleGetItemNode";
  };
  void VisitExpr_(const FunctionNode* fn) final {
    LOG(INFO) << __FUNCTION__<< "FunctionNode";
  }

  void InferDataQuantParam(Expr expr);

  VxOpTable vxOpmap_tbl_;
  IRModule module_;
  GlobalVar var_;
};

std::map<Expr, std::shared_ptr<OpSetup>> MakeTensor(const IRModule& module, const GlobalVar& var,
                                                    const Expr& expr) {
  return TensorMakerImpl(module, var).Create(expr);
}

class GraphMakerImpl : public ExprVisitor{
 public:
  GraphMakerImpl(const IRModule& module, const GlobalVar& var) : module_(module), var_(var) {}

  RawGraphDef Create(const Function& func);

 private:
  void InferCall(const CallNode* cn);
  void VisitInferred(const Expr& expr);
  void VisitExpr_(const CallNode* cn) final;
  void VisitExpr_(const TupleNode* tn) final;
  void VisitExpr_(const TupleGetItemNode* tg) final {
    LOG(INFO) << __FUNCTION__ << "TupleGetItemNode";
  };

  VxOpTable vxOpmap_tbl_;
  std::shared_ptr<tim::vx::Graph> vx_graph_;
  IRModule module_;
  GlobalVar var_;
  static std::shared_ptr<tim::vx::Context> vx_global_ctx_;
};

RawGraphDef MakeGraph(const IRModule& module, const GlobalVar& var, const Function& func) {
  return GraphMakerImpl(module, var).Create(func);
}

class VsiNpuCompiler {
 public:
  static runtime::Module CreateRuntimeModule(const ObjectRef& ref);
};

// TODO: (sven) put env variable to Compiler Option
class VsiNpuCompilerOption {
 private:
  std::string target_platform_name_;
};

runtime::Module CompileVsiNpu(const ObjectRef& ref) {
  return VsiNpuCompiler::CreateRuntimeModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.vsi_npu").set_body_typed(CompileVsiNpu);

}  // namespace vsi_npu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif
