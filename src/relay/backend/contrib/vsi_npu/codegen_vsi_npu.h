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

inline int32_t ConvertAxis(int32_t axisIn, uint32_t dimNum) {
  return dimNum - (axisIn < 0 ? dimNum + axisIn : axisIn) - 1;
}

struct RawGraphDef {
  std::shared_ptr<char> compiled_graph;
  uint32_t compiled_graph_size;
  std::vector<tim::vx::TensorSpec> inputs_spec;
  std::vector<tim::vx::TensorSpec> outputs_spec;
};
class VsiErrorReporter {
 public:
  VsiErrorReporter(const IRModule& module, const GlobalVar& var) : module_(module), var_(var) {}

  void ReportFatalError(const ObjectRef& expr, const VsiError& error) {
    // TODO
  }

 protected:
  tvm::ErrorReporter error_reporter_;
  IRModule module_;
  GlobalVar var_;
};

class TensorMakerImpl : private ExprVisitor, private VsiErrorReporter {
 public:
  TensorMakerImpl(const IRModule& module, const GlobalVar& var) : VsiErrorReporter(module, var) {}

  // std::map<Expr, std::vector<tim::vx::TensorSpec>> Create(const Expr &expr);
  std::map<Expr, std::shared_ptr<OpSetup>> Create(const Expr& expr);

 private:
  void InferCall(const CallNode* cn);
  void VisitInferred(const Expr& expr);

  void VisitExpr_(const CallNode* cn) final;

  // TODO:
  void VisitExpr_(const TupleNode* tn) final;
  void VisitExpr_(const TupleGetItemNode* tg) final {
    std::cout << "TensorMakerImpl: TupleGetItemNode" << std::endl;
  };
  void VisitExpr_(const FunctionNode* fn) final {
    std::cout << "TensorMakerImpl: FunctionNode" << std::endl;
  }

  void InferDataQuantParam(Expr expr);

  VxOpTable vxOpmap_tbl_;
};

std::map<Expr, std::shared_ptr<OpSetup>> MakeTensor(const IRModule& module, const GlobalVar& var,
                                                    const Expr& expr) {
  return TensorMakerImpl(module, var).Create(expr);
}

// class GraphMakerImpl : public MixedModeVisitor, private VsiErrorReporter {
class GraphMakerImpl : public ExprVisitor, private VsiErrorReporter {
 public:
  GraphMakerImpl(const IRModule& module, const GlobalVar& var) : VsiErrorReporter(module, var) {}

  RawGraphDef Create(const Function& func);

 private:
  void InferCall(const CallNode* cn);
  void VisitInferred(const Expr& expr);
  void VisitExpr_(const CallNode* cn) final;
  void VisitExpr_(const TupleNode* tn) final;
  void VisitExpr_(const TupleGetItemNode* tg) final {
    std::cout << "GraphMakerImpl: TupleGetItemNode" << std::endl;
  };
  //   void VisitLeaf(const Expr& expr) final { std::cout << "GraphMakerImpl:
  //   Expr" << std::endl;};

  VxOpTable vxOpmap_tbl_;
  std::shared_ptr<tim::vx::Graph> vx_graph_;

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
