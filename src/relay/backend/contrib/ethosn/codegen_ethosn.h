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
 * \brief The Relay -> Arm(R) Ethos(TM)-N command stream compiler.
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
  std::map<Expr, std::vector<sl::TensorInfo>> Infer(const Expr& expr);

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
  NetworkWithIDs Construct(const Function& func);

 private:
  // Translate from a callnode to the appropriate 'Make' method
  sl::TensorsAndId HandleCall(const CallNode*);

  void VisitExpr_(const CallNode* cn) final;
  void VisitExpr_(const ConstantNode* cn) final;
  void VisitExpr_(const TupleNode* op) final;
  void VisitExpr_(const TupleGetItemNode* tg) final;
  void VisitLeaf(const Expr& expr) final;

  // Make a support library operand from a Call
  EthosnError MakeConvolutionLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeFullyConnectedLayer(const Call&, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeMaxPool2DLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeAvgPool2DLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeReshapeLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeAdditionLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeSigmoidLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeMeanLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeTanhLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeConv2DTransposeLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeConcatenateLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeSplitLayer(const Call& call, sl::TensorsAndId* outs);
  EthosnError MakeDepthToSpaceLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeReluLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeLeakyReLULayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeRequantizeLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeReinterpretQuantizeLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);
  EthosnError MakeResizeLayer(const Call& call, sl::TensorAndId<sl::Operand>* out);

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

/*! \brief Attributes to store the compiler options for Ethos-N */
struct EthosnCompilerConfigNode : public tvm::AttrsNode<EthosnCompilerConfigNode> {
  String variant;
  String sram_size;
  String tops;
  String ple_ratio;
  bool strategy0;
  bool strategy1;
  bool strategy3;
  bool strategy4;
  bool strategy6;
  bool strategy7;
  bool dump_ram;
  bool initial_sram_dump;
  bool block_config_16x16;
  bool block_config_32x8;
  bool block_config_8x32;
  bool block_config_8x8;
  bool enable_intermediate_compression;
  bool disable_winograd;
  String debug_dir;
  bool inline_non_compute_intensive_partitions;
  bool experimental_compiler;

  TVM_DECLARE_ATTRS(EthosnCompilerConfigNode, "ext.attrs.EthosnCompilerConfigNode") {
    TVM_ATTR_FIELD(variant).describe("See Ethos-N documentation.").set_default("n78");
    TVM_ATTR_FIELD(sram_size)
        .describe("Optionally override the default sram size. See Ethos(TM)-N documentation.")
        .set_default("0");
    TVM_ATTR_FIELD(tops)
        .describe("Valid values 1, 2, 4 and 8. See Ethos(TM)-N documentation.")
        .set_default("1");
    TVM_ATTR_FIELD(ple_ratio)
        .describe("Valid values 2 and 4. See Ethos(TM)-N documentation.")
        .set_default("2");
    TVM_ATTR_FIELD(strategy0).set_default(true);
    TVM_ATTR_FIELD(strategy1).set_default(true);
    TVM_ATTR_FIELD(strategy3).set_default(true);
    TVM_ATTR_FIELD(strategy4).set_default(true);
    TVM_ATTR_FIELD(strategy6).set_default(true);
    TVM_ATTR_FIELD(strategy7).set_default(true);
    TVM_ATTR_FIELD(dump_ram).set_default(false);
    TVM_ATTR_FIELD(initial_sram_dump).set_default(false);
    TVM_ATTR_FIELD(block_config_16x16).set_default(true);
    TVM_ATTR_FIELD(block_config_32x8).set_default(true);
    TVM_ATTR_FIELD(block_config_8x32).set_default(true);
    TVM_ATTR_FIELD(block_config_8x8).set_default(true);
    TVM_ATTR_FIELD(enable_intermediate_compression).set_default(true);
    TVM_ATTR_FIELD(disable_winograd).set_default(false);
    TVM_ATTR_FIELD(debug_dir).set_default(".");
    TVM_ATTR_FIELD(inline_non_compute_intensive_partitions)
        .describe(
            "A heuristic to improve performance. Inlines functions partitioned for Arm(R) "
            "Ethos(TM)-N that are deemed 'non-compute-intensive'. The inlined functions will "
            "continue through TVM's standard compilation flow.")
        .set_default(true);
    TVM_ATTR_FIELD(experimental_compiler)
        .describe("An exprimental cascading compiler for Arm(R) Ethos(TM)-N.")
        .set_default(false);
  }
};

class EthosnCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(EthosnCompilerConfig, Attrs, EthosnCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(EthosnCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.ethos-n.options", EthosnCompilerConfig);

EthosnCompilerConfig GetCompilerAttrs() {
  auto ctx = transform::PassContext::Current();
  Optional<EthosnCompilerConfig> cfg =
      ctx->GetConfig<EthosnCompilerConfig>("relay.ext.ethos-n.options");
  if (!cfg.defined()) {
    return AttrsWithDefaultValues<EthosnCompilerConfig>();
  }
  return cfg.value();
}
TVM_REGISTER_GLOBAL("relay.ext.ethos-n.get_compiler_attrs").set_body_typed(GetCompilerAttrs);

/*! \brief The compiler for Ethos-N functions */
class EthosnCompiler {
 public:
  /*!
   * \brief Create an Ethos-N runtime module from a Relay Ethos-N function
   * \param ref An ObjectRef pointing to a Relay Ethos-N function
   * \return runtime_module An Ethos-N runtime module
   */
  static runtime::Module CreateRuntimeModule(const ObjectRef& ref);

  /*!
   * \brief Initialise the is-supported functionality of the Ethos-N support library
   * with the target variant.
   * \return Error object
   */
  static EthosnError SupportedSetup();

  /*!
   * \brief Return the is-supported API of the Support Library
   * \return A reference to the API.
   */
  static std::unique_ptr<sl::SupportQueries>& GetSupported() {
    ICHECK(m_Queries != nullptr);
    return m_Queries;
  }

 private:
  /*!
   * \brief Compile a single Relay Ethos-N function into an ordered compiled network.
   * Compilation options will be taken from the PassContext.
   * \param mod The module the function is stored in (for error reporting purposes)
   * \param gvar The global var corresponding to the function
   * \param func The function to be compiled
   * \return ordered_compiled_network A compiled network with additional information
   * to handle difference in input/output ordering between the TVM runtime and the
   * Ethos-N compiled network.
   */
  static runtime::ethosn::OrderedCompiledNetwork CompileEthosnFunc(const IRModule& mod,
                                                                   const GlobalVar& gvar,
                                                                   const Function& func);

  /*!
   * \brief Get the Support Library compilation options from the PassContext
   * \return options The compilation options
   */
  static sl::CompilationOptions CreateOptions();

  /*!
   * \brief Determine the order in which inputs should be provided/outputs should be
   * read from a compiled network. This is required because when you compile a network
   * for Ethos-N, you don't have control over the order in which the inputs/outputs
   * are given. You can, however, query what order the compiler decided to give them in.
   * We therefore keep track of our desired order and the actual order and create a
   * small translation table between the two for use in the runtime.
   * \param network A network additionally with the desired input/output order
   * \param compiled_network The compiled network with an as yet undetermined input/output order
   * \return input_output_order The order in which to permute the inputs/outputs given
   * by the TVM runtime such that they map correctly to the compiled network.
   */
  static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> GetInputOutputOrder(
      NetworkWithIDs network, const std::unique_ptr<sl::CompiledNetwork>& compiled_network);

  /*!
   * \brief Determine the input and output sizes of a compiled network.
   *
   * These need to be queried from the compiled network as the compiler can choose
   * to add additional padding on the input/output in certain cases.
   *
   * \param compiled_network The network compiled by the NPU compiler.
   * \return Pair of vectors of buffer sizes for both the inputs and outputs of the
   * network.
   */
  static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> GetIOSizes(
      const std::unique_ptr<sl::CompiledNetwork>& compiled_network);

  /*!
   * \brief Query interface used to determine if the Ethos-N hardware supports an operation
   * with the supplied parameters.
   */
  static std::unique_ptr<sl::SupportQueries> m_Queries;
};

runtime::Module CompileEthosn(const ObjectRef& ref) {
  return EthosnCompiler::CreateRuntimeModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.ethos-n").set_body_typed(CompileEthosn);

TVM_REGISTER_GLOBAL("relay.ext.ethos-n.constant_updater")
    .set_body_typed([](Expr expr, std::string symbol) { return Map<String, runtime::NDArray>(); });

}  // namespace ethosn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ETHOSN_CODEGEN_ETHOSN_H_
