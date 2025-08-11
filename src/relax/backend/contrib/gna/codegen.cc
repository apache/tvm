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
 * \file src/relax/backend/contrib/gna/codegen.cc
 * \brief Implementation of the GNA JSON serializer.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/expr.h>

#include <string>

#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONSerializer = backend::contrib::JSONSerializer;
using backend::contrib::NodeEntries;

class GNAJSONSerializer : public JSONSerializer {
 public:
  GNAJSONSerializer(Map<Constant, String> constant_names, Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  NodeEntries VisitExpr_(const CallNode* call_node) final {
    const auto* fn_var = call_node->op.as<VarNode>();
    ICHECK(fn_var);
    const auto fn = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr<String>(attr::kComposite);
    ICHECK(composite_opt.has_value()) << "Only composite functions are supported.";

    std::string composite_name = composite_opt.value();

    NodeEntries inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);

    const CallNode* root_call = nullptr;
    if (composite_name.find("gna.dense") != std::string::npos) {
      root_call = backend::GetOpInFunction(fn, "relax.matmul");
    } else if (composite_name.find("gna.conv1d") != std::string::npos) {
      root_call = backend::GetOpInFunction(fn, "relax.nn.conv1d");
    } else if (composite_name.find("gna.relu") != std::string::npos) {
      root_call = backend::GetOpInFunction(fn, "relax.nn.relu");
    } else {
      LOG(FATAL) << "Unimplemented GNA pattern: " << composite_name;
    }

    SetCallNodeAttribute(node, root_call);
    return AddNode(node, GetRef<Expr>(call_node));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  Map<Var, Expr> bindings_;

  void SetCallNodeAttribute(std::shared_ptr<JSONGraphNode> node, const CallNode* call) {
    // First call the base implementation to extract standard attributes
    JSONSerializer::SetCallNodeAttribute(node, call);

    // Add GNA-specific attributes based on the operation
    if (call && call->op.as<OpNode>()) {
      auto op = Downcast<Op>(call->op);
      std::string op_name = op->name;

      // Extract shape information from struct_info
      if (!call->args.empty()) {
        StructInfo input_sinfo = GetStructInfo(call->args[0]);
        if (const auto* tensor_sinfo = input_sinfo.as<TensorStructInfoNode>()) {
          if (tensor_sinfo->shape.defined()) {
            std::vector<std::string> shape_strs;
            ShapeExpr shape = Downcast<ShapeExpr>(tensor_sinfo->shape.value());
            for (const PrimExpr& dim : shape->values) {
              if (const auto* int_imm = dim.as<tvm::tir::IntImmNode>()) {
                shape_strs.push_back(std::to_string(int_imm->value));
              } else {
                shape_strs.push_back("-1");
              }
            }
            std::vector<dmlc::any> shape_attr;
            shape_attr.emplace_back(shape_strs);
            node->SetAttr("input_shape", shape_attr);
          }

          std::vector<std::string> dtype_strs{tensor_sinfo->dtype.code() == kDLFloat ? "float32"
                                                                                     : "int32"};
          std::vector<dmlc::any> dtype_attr;
          dtype_attr.emplace_back(dtype_strs);
          node->SetAttr("input_dtype", dtype_attr);
        }
      }

      if (op_name == "relax.nn.conv1d") {
        if (call->attrs.defined()) {
          std::vector<std::string> op_attrs{"conv1d_op"};
          std::vector<dmlc::any> op_attr;
          op_attr.emplace_back(op_attrs);
          node->SetAttr("gna_op_type", op_attr);
        }
      } else if (op_name == "relax.matmul") {
        std::vector<std::string> op_attrs{"dense_op"};
        std::vector<dmlc::any> op_attr;
        op_attr.emplace_back(op_attrs);
        node->SetAttr("gna_op_type", op_attr);
      } else if (op_name == "relax.nn.relu") {
        std::vector<std::string> op_attrs{"activation_op"};
        std::vector<dmlc::any> op_attr;
        op_attr.emplace_back(op_attrs);
        node->SetAttr("gna_op_type", op_attr);
      }
    }
  }
};

/*!
 * \brief Create a GNA JSON runtime module.
 * \param functions The functions to be compiled.
 * \param unused Unused config options.
 * \param constant_names The constant names to be used.
 * \return Array of runtime modules.
 */
Array<runtime::Module> GNACompiler(Array<Function> functions, Map<String, ffi::Any> /*unused*/,
                                   Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;

  for (const auto& func : functions) {
    GNAJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto constant_names_used = serializer.GetConstantNames();

    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.GNAJSONRuntimeCreate");
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back(
        pf(func_name, graph_json, constant_names_used).cast<runtime::Module>());
  }

  return compiled_functions;
}

// Register the external codegen entrypoint via FFI reflection (new TVM registry)
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.gna", GNACompiler);
});

}  // namespace contrib
}  // namespace relax

namespace target {

// Register GNA target kind
TVM_REGISTER_TARGET_KIND("gna", kDLExtDev).set_default_keys({"gna"});

}  // namespace target

}  // namespace tvm
