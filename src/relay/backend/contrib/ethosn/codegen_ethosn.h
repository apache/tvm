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
 * \file src/relay/backend/contrib/ethosn/codegen_ethosn.h
 * \brief The Relay -> Ethos-N command stream compiler.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_ETHOSN_CODEGEN_ETHOSN_H_
#define TVM_RELAY_BACKEND_CONTRIB_ETHOSN_CODEGEN_ETHOSN_H_

#include <dmlc/memory_io.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../../runtime/contrib/ethosn/ethosn_runtime.h"
#include "../codegen_c/codegen_c.h"
#include "ethosn_api.h"
#include "ethosn_support_library/Support.hpp"
#include "ethosn_support_library/SupportQueries.hpp"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosn {

namespace sl = ::ethosn::support_library;

/*!
 * \brief A struct to hold an uncompiled support library network alongside
 * the desired order of input and output operation ids.
 */
struct NetworkWithIDs {
  struct hash_pair {
    template <class T_0, class T_1>
    size_t operator()(const std::pair<T_0, T_1>& p) const {
      return std::hash<T_0>{}(p.first) ^ std::hash<T_1>{}(p.second);
    }
  };
  std::shared_ptr<sl::Network> network;
  std::unordered_map<uint32_t, unsigned int> input_ids;
  std::unordered_map<std::pair<uint32_t, uint32_t>, unsigned int, hash_pair> output_ids;
};

/*!
 * \brief A base class for error handling using ErrorReporter.
 */
class ErrorReportingPass {
 public:
  ErrorReportingPass(const IRModule& mod, const GlobalVar& var) : mod_(mod), var_(var) {}

  /*!
   * \brief Report fatal errors for an expression.
   * \param expr The expression to report errors at.
   * \param err The errors to report.
   */
  void ReportFatalError(const ObjectRef& expr, const EthosnError& err) {
    for (const auto& msg : err.msgs) {
      error_reporter_.ReportAt(this->var_, expr, ErrorBuilder() << msg);
    }
    error_reporter_.RenderErrors(this->mod_);
  }

 protected:
  /*! \brief An ErrorReporter object to render the errors.*/
  ErrorReporter error_reporter_;
  /*! \brief The module to report errors for. */
  IRModule mod_;
  /*! \brief The GlobalVar to report errors for. */
  GlobalVar var_;
};

/*!
 * \brief A custom pass to infer the support library tensor information
 * for a Relay expression.
 *
 * Support Library requires that tensors are explicitly declared with
 * information on their size, data type, format (eg. NHWC) and quantisation
 * parameters. In Relay, size and data type are already determined when the
 * type_infer pass is run. However, format and quantisation parameters are
 * properties of the operators that consume the tensors.
 *
 * This pass works by having each node initialise the information of its
 * parents, essentially propagating the inferred information all the way up
 * to the inputs of the expression.
 *
 * Because the children initialise the information of the parents, it is
 * necessary to traverse the graph in such a way so as to ensure all the
 * children of a node are visited before the parent is. As Relay does not
 * keep a reference to child nodes, this pass goes in preorder but will
 * skip visiting a parent if all the children haven't yet been visited (see
 * VisitInferred for the logic that implements this).
 *
 * Inference only works for supported callnodes, for tuplenodes, tuplegetitem
 * nodes and free var nodes. Other nodes should not be off-loaded to Ethos-N.
 */
class InferTensorsVisitor : private ErrorReportingPass, private ExprVisitor {
 public:
  InferTensorsVisitor(const IRModule& mod, const GlobalVar& var) : ErrorReportingPass(mod, var) {}

  /*!
   * \brief Infer the support library tensor information for all the nodes
   * in an expression.
   * \param expr The expression for which to infer tensor information.
   * \return A map of expressions to tensor information.
   * \note This algorithm does not traverse into functions, so call it on
   * the body of the function you're interested in.
   */
  std::map<Expr, std::vector<sl::TensorInfo>> Infer(const Expr& expr) {
    tensor_table_.clear();
    CHECK(expr->checked_type().defined());
    size_t output_size = 1;
    if (expr->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type = expr->checked_type().as<TupleTypeNode>();
      output_size = type->fields.size();
    }
    for (size_t i = 0; i < output_size; i++) {
      tensor_table_[expr].push_back(sl::TensorInfo({1, 1, 1, 1}, sl::DataType::UINT8_QUANTIZED,
                                                   sl::DataFormat::NHWC, sl::QuantizationInfo()));
    }
    VisitInferred(expr);
    return tensor_table_;
  }

 private:
  // Infer a callnode if it's a supported operator/composite function
  void InferCall(const CallNode* cn);
  void VisitInferred(const Expr& expr);

  void VisitExpr_(const CallNode* cn) final;
  void VisitExpr_(const TupleNode* tn) final;
  void VisitExpr_(const TupleGetItemNode* tg) final;
  // Don't traverse into functions, the Ethos-N codegen isn't meant to support them.
  void VisitExpr_(const FunctionNode* fn) final {}

  /*! \brief A look-up table from Expr to tensor infos. */
  std::map<Expr, std::vector<sl::TensorInfo>> tensor_table_;
};

std::map<Expr, std::vector<sl::TensorInfo>> InferTensors(const IRModule& mod, const GlobalVar& var,
                                                         const Expr& expr) {
  return InferTensorsVisitor(mod, var).Infer(expr);
}

/*!
 * \brief A pass to generate a support library network from a Relay function.
 *
 * This pass constructs an equivalent support library network from a Relay
 * function in two visits. One to infer the tensor information of all the nodes
 * and another in postorder to add the nodes as support library operands.
 * (Supported) Callnodes, tuplenodes, tuplegetitemnodes and (free)
 * varnodes are handled by this pass.
 *
 * As part of the pass, nodes in the function body are associated with both
 * type information in the 'tensor_table', and support library operands in the
 * 'operand_table'. Both of these are maps of vectors as a Relay node can have
 * tuple type and accordingly be associated with multiple tensors. For nodes
 * which are not tuple type, vectors of size 1 are used.
 */
class ConstructNetworkVisitor : public MixedModeVisitor, private ErrorReportingPass {
 public:
  explicit ConstructNetworkVisitor(const IRModule& mod, const GlobalVar& var)
      : ErrorReportingPass(mod, var) {}

  /*!
   * \brief Construct a support library network from a given Relay function. The
   * function should contain only nodes supported by Ethos-N.
   * \param func The Relay function for which to construct a support library network.
   * \return A support library network that performs the same operation as the Relay
   * function.
   */
  NetworkWithIDs Construct(const Function& func) {
    // Initialise everything
    NetworkWithIDs network_with_ids;
    network_ = sl::CreateNetwork();
    network_with_ids.network = network_;
    operand_table_.clear();

    // Infer tensor information
    tensor_table_ = InferTensors(this->mod_, this->var_, func->body);
    // Add the inputs in the order they appear in the parameters
    unsigned int idx = 0;
    for (const auto& param : func->params) {
      for (const auto& tensor_info : tensor_table_[param]) {
        auto tensor_and_id = AddInput(network_, tensor_info);
        operand_table_[param].push_back(tensor_and_id.tensor);
        id_table_[param].push_back(std::make_pair(tensor_and_id.operationId, 0));
        network_with_ids.input_ids[tensor_and_id.operationId] = idx++;
      }
    }
    // Add the function body
    VisitExpr(func->body);
    // Add the outputs
    idx = 0;
    for (const auto& layer : operand_table_[func->body]) {
      AddOutput(network_, *layer);
      network_with_ids.output_ids[id_table_[func->body][idx]] = idx;
      idx++;
    }
    return network_with_ids;
  }

 private:
  // Translate from a callnode to the appropriate 'Make' method
  sl::TensorsAndId HandleCall(const CallNode*);

  void VisitExpr_(const CallNode* cn) final;
  void VisitExpr_(const TupleNode* op) final;
  void VisitExpr_(const TupleGetItemNode* tg) final;
  void VisitLeaf(const Expr& expr) final;

  // Make a support library operand from a Call
  EthosnError MakeConcatenateLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeSplitLayer(const Call& call, sl::TensorsAndId* outs);

  /*! \brief A look-up table from Expr to layers. */
  std::map<Expr, std::vector<std::shared_ptr<sl::Operand>>> operand_table_;
  /*! \brief A look-up table from Expr to SL operation IDs. */
  std::map<Expr, std::vector<std::pair<uint32_t, uint32_t>>> id_table_;
  /*! \brief A look-up table from Expr to tensor infos. */
  std::map<Expr, std::vector<sl::TensorInfo>> tensor_table_;
  /*! \brief The support library network to compile. */
  std::shared_ptr<sl::Network> network_;
};

NetworkWithIDs ConstructNetwork(const IRModule& mod, const GlobalVar& var, const Function& func) {
  return ConstructNetworkVisitor(mod, var).Construct(func);
}

class EthosnCompiler {
 public:
  static runtime::ethosn::OrderedCompiledNetwork CompileEthosnFunc(const IRModule& mod,
                                                                   std::string name,
                                                                   const Function& func) {
    // Construct the network
    GlobalVar var = mod->GetGlobalVar(name);
    auto network_with_ids = ConstructNetwork(mod, var, func);
    // Now set the required build flags
    sl::CompilationOptions options = EthosnAPI::CreateOptions();
    // Finally compile the network
    auto compiled_network = EthosnAPI::Compile(network_with_ids.network, options);
    auto input_output_order = GetInputOutputOrder(network_with_ids, compiled_network);
    runtime::ethosn::OrderedCompiledNetwork ordered_network;
    ordered_network.name = name;
    ordered_network.cmm = std::move(compiled_network);
    ordered_network.inputs = input_output_order.first;
    ordered_network.outputs = input_output_order.second;
    return ordered_network;
  }

  static runtime::Module CreateRuntimeModule(const ObjectRef& ref) {
    std::vector<runtime::ethosn::OrderedCompiledNetwork> cmms;
    if (ref->IsInstance<FunctionNode>()) {
      IRModule mod;
      Function bfunc = Downcast<Function>(ref);
      auto name_node = bfunc->GetAttr<String>(tvm::attr::kGlobalSymbol);
      CHECK(name_node.defined()) << "Failed to retrieved external symbol.";
      mod->Add(GlobalVar(name_node.value()), bfunc);
      for (const auto& it : mod->functions) {
        Function func = Downcast<Function>(it.second);
        name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        cmms.emplace_back(CompileEthosnFunc(mod, name_node.value(), func));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function";
    }
    auto n = make_object<runtime::ethosn::EthosnModule>(&cmms);
    return runtime::Module(n);
  }

  static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> GetInputOutputOrder(
      NetworkWithIDs network, const std::unique_ptr<sl::CompiledNetwork>& compiled_network) {
    std::vector<sl::InputBufferInfo> input_infos = compiled_network->GetInputBufferInfos();
    std::vector<sl::OutputBufferInfo> output_infos = compiled_network->GetOutputBufferInfos();
    std::vector<uint32_t> input_order;
    std::vector<uint32_t> output_order;
    for (const auto& input_info : input_infos) {
      input_order.push_back(network.input_ids[input_info.m_SourceOperationId]);
    }
    for (const auto& output_info : output_infos) {
      auto output_id =
          std::make_pair(output_info.m_SourceOperationId, output_info.m_SourceOperationOutputIndex);
      output_order.push_back(network.output_ids[output_id]);
    }
    return std::make_pair(input_order, output_order);
  }
};

runtime::Module CompileEthosn(const ObjectRef& ref) {
  return EthosnCompiler::CreateRuntimeModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.ethos-n").set_body_typed(CompileEthosn);

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_CODEGEN_ETHOSN_H_
