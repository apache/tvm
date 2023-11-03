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
 * \file relay/backend/graph_codegen.cc
 * \brief Graph executor codegen
 */

#include <dmlc/any.h>
#include <dmlc/json.h>
#include <tvm/ir/module.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>

#include <list>
#include <string>
#include <vector>

#include "../op/annotation/annotation.h"
#include "../op/call/call.h"
#include "../op/memory/device_copy.h"
#include "../transforms/device_aware_visitors.h"
#include "./te_compiler.h"
#include "./utils.h"

namespace tvm {
namespace relay {

// TODO(@jroesch, @csullivan): declare directly elsewhere
backend::StaticMemoryPlan GraphPlanMemory(const Function& func);

namespace backend {

class GraphNode;
class GraphInputNode;
class GraphOpNode;

using IntegerArray = Array<Integer>;
using ShapeVector = std::vector<std::vector<int64_t>>;
using GraphAttrs = std::unordered_map<std::string, dmlc::any>;
using GraphObjectPtr = std::shared_ptr<GraphNode>;
using GraphInputObjectPtr = std::shared_ptr<GraphInputNode>;
using GraphOpObjectPtr = std::shared_ptr<GraphOpNode>;

/*! \brief Node types */
enum GraphNodeType {
  kGraphNop,
  kGraphInputNode,
  kGraphOpNode,
};

class GraphNodeRef {
 public:
  GraphNodeRef() {}
  GraphNodeRef(int ident, int index, int version = 0)
      : ident_(ident), index_(index), version_(version) {}

  inline void Save(dmlc::JSONWriter* writer) const {
    writer->BeginArray();
    writer->WriteArrayItem(ident_);
    writer->WriteArrayItem(index_);
    writer->WriteArrayItem(version_);
    writer->EndArray();
  }

  inline void Load(dmlc::JSONReader* reader) { LOG(FATAL) << "Not implemented."; }

 protected:
  int ident_;
  int index_{0};
  int version_{0};
};

/*! \brief Base Node class */
class GraphNode {
 public:
  GraphNode() {}
  virtual void Save(dmlc::JSONWriter* writer) const {}
  virtual void Load(dmlc::JSONReader* reader) {}
  virtual GraphNodeType Type() const { return kGraphNop; }
  virtual ~GraphNode() {}

 public:
  int num_outputs_{1};
  std::string name_;
  GraphAttrs attrs_;
};

/*! \brief Input Node */
class GraphInputNode : public GraphNode {
 public:
  GraphInputNode() {}
  GraphInputNode(const std::string& name, const GraphAttrs& attrs) {
    name_ = name;
    attrs_ = attrs;
  }

  GraphNodeType Type() const override { return kGraphInputNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    const std::string op_name{"null"};
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_name);
    writer->WriteObjectKeyValue("name", this->name_);
    writer->WriteObjectKeyValue("inputs", std::list<int>());
    writer->EndObject();
  }
  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& attrs) {
    auto ptr = std::make_shared<GraphInputNode>(name, attrs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }
};

/*! \brief Op Node */
class GraphOpNode : public GraphNode {
 public:
  GraphOpNode() {}
  GraphOpNode(const std::string& name, const GraphAttrs& nd_attrs, const std::string& op_name,
              const std::vector<GraphNodeRef>& inputs, const GraphAttrs& attrs,
              size_t num_outputs = 1) {
    name_ = name;
    attrs_ = nd_attrs;
    op_name_ = op_name;
    inputs_ = inputs;
    op_attrs_ = attrs;
    num_outputs_ = num_outputs;
    op_attrs_["func_name"] = op_name_;
    op_attrs_["flatten_data"] = std::string("0");
    op_attrs_["num_inputs"] = std::to_string(inputs_.size());
    op_attrs_["num_outputs"] = std::to_string(num_outputs_);
  }

  GraphNodeType Type() const override { return kGraphOpNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    GraphAttrs attrs = op_attrs_;
    attrs["func_name"] = this->op_name_;
    attrs["flatten_data"] = std::string("0");
    attrs["num_inputs"] = std::to_string(this->inputs_.size());
    attrs["num_outputs"] = std::to_string(this->num_outputs_);
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_type_name_);
    writer->WriteObjectKeyValue("name", name_);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("inputs", this->inputs_);
    writer->EndObject();
  }
  static std::shared_ptr<GraphNode> make_node_ptr(const std::string& name,
                                                  const GraphAttrs& nd_attrs,
                                                  const std::string& op_name,
                                                  const std::vector<GraphNodeRef>& inputs,
                                                  const GraphAttrs& attrs, size_t num_outputs = 1) {
    auto ptr = std::make_shared<GraphOpNode>(name, nd_attrs, op_name, inputs, attrs, num_outputs);
    return std::dynamic_pointer_cast<GraphNode>(ptr);
  }

 public:
  std::string op_name_;
  std::vector<GraphNodeRef> inputs_;
  GraphAttrs op_attrs_;

 private:
  const std::string op_type_name_{"tvm_op"};
};

/*! \brief Code generator for the graph executor, produces a module containing the graph JSON,
 * module, and parameters.
 */
class GraphExecutorCodegen : public backend::MemoizedExprTranslator<std::vector<GraphNodeRef>> {
 public:
  GraphExecutorCodegen(runtime::Module* mod, const Array<Target>& targets)
      : mod_(mod), config_(transform::PassContext::Current(), targets) {}

  StorageInfo GetStorageInfo(const Expr& e) {
    size_t count = memory_plan_->expr_to_storage_info.count(e);
    ICHECK_GT(count, 0) << "Expr is not existing in storage plan";
    auto storage_info = memory_plan_->expr_to_storage_info[e];
    return storage_info;
  }

  LoweredOutput Codegen(IRModule mod, relay::Function func, String mod_name) {
    mod_name_ = mod_name;
    VLOG_CONTEXT << "GraphExecutorCodegen";
    VLOG(1) << "compiling:" << std::endl << PrettyPrint(func);

    // TODO(mbs): Why plan memory and update workspace sizes before lowering?
    memory_plan_ = GraphPlanMemory(func);

    backend::FunctionInfo func_info;

    if (memory_plan_.defined()) {
      // TODO(@electriclilies, @jroesch): remove UpdateMainWorkspaceSize
      func_info =
          relay::tec::UpdateMainWorkspaceSize(mod, config_, memory_plan_->expr_to_storage_info);
      mod = WithAttr(mod, "main_func_info", func_info);
    }

    IRModule lowered_mod = tec::LowerTE(mod_name_, config_, [this](BaseFunc func) {
      // We need to maintain the constant map for external
      // functions so we pass this processing function which
      // allows us to process each function as we lower it.
      if (func->GetAttr<String>(attr::kCompiler).defined()) {
        UpdateConstants(func, &params_);
      }

      // TODO(@areusch, @jroesch): We should refactor this to
      // execute as a further pass, instead writing data to the
      // lowering process directly.
      tec::UpdateFunctionMetadata(func, this->function_metadata_);
    })(mod);

    Optional<backend::FunctionInfo> main_func_info =
        lowered_mod->GetAttr<backend::FunctionInfo>("main_func_info");

    function_metadata_.Set(runtime::symbol::tvm_module_main, main_func_info.value());

    Function lowered_main_func = Downcast<Function>(lowered_mod->Lookup("main"));

    // Now that we have lowered all operators to TIR code, we can proceed with compilation.
    //
    // We need to unfortunately re-plan as the previous results have been invalidated by lowering
    // we will fix this in future refactors.
    memory_plan_ = GraphPlanMemory(lowered_main_func);

    // The graph planner also can not handle planning calls to global variables to we must remap

    // First we convert all the parameters into input nodes.
    for (auto param : lowered_main_func->params) {
      auto node_ptr = GraphInputNode::make_node_ptr(param->name_hint(), GraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }

    heads_ = VisitExpr(lowered_main_func->body);
    std::ostringstream os;

    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();

    // Collect any runtime modules generated by external codegen.
    ret.external_mods =
        lowered_mod->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});

    // Collect any constants extracted by external codegen.
    ret.params = std::unordered_map<std::string, tvm::runtime::NDArray>();
    Map<String, runtime::NDArray> const_name_to_constant =
        lowered_mod->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant)
            .value_or({});
    for (const auto& kv : const_name_to_constant) {
      VLOG(1) << "constant '" << kv.first << "' contributed by external codegen";
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    // Collect any constants extracted during lowering.
    for (const auto& kv : params_) {
      VLOG(1) << "constant '" << kv.first << "' contributed by TECompiler";
      ICHECK(ret.params.emplace(kv.first, kv.second).second);
    }

    ret.function_metadata = std::move(function_metadata_);

    // This is the point where we separate the functions in the module by target
    ret.lowered_funcs = tec::GetPerTargetModules(lowered_mod);
    ret.metadata =
        ExecutorCodegenMetadata({} /* inputs */, {} /* input_tensor_types */, {} /* outputs */,
                                {} /* output_tensor_types */, {} /* pools */, {} /* devices */,
                                runtime::kTvmExecutorGraph /* executor */, mod_name_ /* mod_name */,
                                "packed" /* interface_api */, Bool(false) /* unpacked_api */);
    return ret;
  }

 protected:
  /*!
   * \brief Add node to graph
   *
   * \param node
   * \param expr
   * \return std::vector<_NodeRef>
   */
  std::vector<GraphNodeRef> AddNode(GraphObjectPtr node, Expr expr) {
    auto checked_type = expr->checked_type();

    auto storage_info = GetStorageInfo(expr);
    // storage
    std::vector<int64_t> storage_ids;
    for (auto v : storage_info->storage_ids) {
      storage_ids.push_back(v);
    }
    node->attrs_["storage_id"] = std::move(storage_ids);
    // type
    std::vector<int64_t> device_types;
    for (const auto& virtual_device : storage_info->virtual_devices) {
      // TODO(mbs): Keeping only the device type.
      ICHECK_GT(virtual_device->device_type(), 0);
      device_types.push_back(virtual_device->device_type());
    }
    size_t num_unknown_devices = std::count(device_types.begin(), device_types.end(), 0);
    if (num_unknown_devices != 0 && num_unknown_devices != device_types.size()) {
      LOG(FATAL) << "The graph contains not annotated nodes for "
                 << "heterogeneous execution. All nodes must be "
                 << "annotated.";
    }
    if (num_unknown_devices == 0) {
      node->attrs_["device_index"] = device_types;
    }
    // storage scope
    std::vector<std::string> storage_scope;
    for (const auto& virtual_device : storage_info->virtual_devices) {
      storage_scope.push_back(std::string(virtual_device->memory_scope));
    }
    node->attrs_["storage_scope"] = std::move(storage_scope);
    auto node_id = nodes_.size();
    nodes_.push_back(node);
    // Tuple return value, flatten as tuple
    if (const auto* tuple_type = checked_type.as<TupleTypeNode>()) {
      std::vector<GraphNodeRef> ret;
      ShapeVector shape;
      std::vector<std::string> dtype;
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        if (const auto* typ = tuple_type->fields[i].as<TensorTypeNode>()) {
          ret.push_back(GraphNodeRef(node_id, i));
          shape.emplace_back(ShapeToJSON(typ->shape));
          dtype.emplace_back(DType2String(typ->dtype));
        } else {
          LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
        }
      }
      ICHECK_EQ(node->Type(), kGraphOpNode);
      auto op_nd = std::dynamic_pointer_cast<GraphOpNode>(node);
      op_nd->attrs_["shape"] = shape;
      op_nd->attrs_["dtype"] = dtype;
      op_nd->num_outputs_ = tuple_type->fields.size();
      return ret;
    }
    // Normal tensor return type
    if (const auto* tensor_type = checked_type.as<TensorTypeNode>()) {
      ShapeVector shape;
      std::vector<std::string> dtype;
      shape.emplace_back(ShapeToJSON(tensor_type->shape));
      dtype.emplace_back(DType2String(tensor_type->dtype));
      node->attrs_["shape"] = shape;
      node->attrs_["dtype"] = dtype;
    } else {
      LOG(FATAL) << "type " << checked_type->GetTypeKey() << " not supported";
    }
    return {GraphNodeRef(node_id, 0)};
  }

  std::vector<GraphNodeRef> VisitExpr_(const VarNode* op) override {
    Expr expr = GetRef<Expr>(op);
    return var_map_[expr.get()];
  }

  std::vector<GraphNodeRef> VisitExpr_(const ConstantNode* op) override {
    Expr expr = GetRef<Expr>(op);
    size_t index = params_.size();
    std::string name = "p" + std::to_string(index);
    auto node = GraphInputNode::make_node_ptr(name, GraphAttrs());
    auto to_return = AddNode(node, expr);
    CHECK_EQ(to_return.size(), 1) << "Expected exactly 1 parameter node created";
    param_storage_ids_[name] = GetStorageInfo(expr)->storage_ids[0];
    params_[name] = op->data;
    return to_return;
  }

  std::vector<GraphNodeRef> VisitExpr_(const TupleNode* op) override {
    std::vector<GraphNodeRef> fields;
    for (auto field : op->fields) {
      auto ref_vec = VisitExpr(field);
      for (auto ref : ref_vec) {
        fields.push_back(ref);
      }
    }
    return fields;
  }

  bool ShareSameStorage(const Expr& lhs, const Expr& rhs) {
    StorageInfo lit = GetStorageInfo(lhs);
    StorageInfo rit = GetStorageInfo(rhs);
    int64_t lhs_storage_id = lit->storage_ids[0];
    int64_t rhs_storage_id = rit->storage_ids[0];
    return lhs_storage_id == rhs_storage_id;
  }

  std::vector<GraphNodeRef> GraphAddCallNode(const CallNode* call_node, GraphAttrs attrs) {
    Call call = GetRef<Call>(call_node);
    std::vector<GraphNodeRef> inputs;
    std::string func_name;

    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);
    if (device_copy_props.body.defined()) {
      // The graph executor expects to see a normal call to the undefined @__copy function.
      // The source and destination device annotations are no longer needed since they have
      // been captured in the StorageInfos for both input and output.
      // TODO(mbs): device_copy cleanup
      func_name = "__copy";
      for (const auto& n : VisitExpr(device_copy_props.body)) {
        inputs.push_back(n);
      }
    } else if (call_lowered_props.lowered_func.defined()) {
      // Extract function and arguments from the call_lowered op

      func_name = call_lowered_props.lowered_func->name_hint;

      for (const Expr& arg : call_lowered_props.arguments) {
        for (auto n : VisitExpr(arg)) {
          inputs.push_back(n);
        }
      }
      if (call_lowered_props.attrs.metadata.count("relay_attrs")) {
        if (auto relay_attrs =
                call_lowered_props.attrs.metadata["relay_attrs"].as<DictAttrsNode>()) {
          for (auto p : relay_attrs->dict) {
            if (p.second.as<StringObj>()) {
              attrs[p.first] = std::string(Downcast<String>(p.second));
            }
          }
        }
      }
      // TODO(mbs): "reshape" cleanup.
      if (IsReshapeOnly(call_lowered_props) &&
          ShareSameStorage(GetRef<Expr>(call_node), call_lowered_props.arguments[0])) {
        auto node = GraphOpNode::make_node_ptr("reshape_nop", GraphAttrs(), "__nop", inputs, attrs);
        return AddNode(node, call);
      }
    } else if (!call_node->attrs.defined()) {  // Call is an extern function
      const auto* func = call_node->op.as<GlobalVarNode>();
      ICHECK(func) << "Expected the operator to be a global var, but got "
                   << call_node->op->GetTypeKey();  // getting a relay fn here, not sure why.
      func_name = func->name_hint;

      for (const Expr& arg : call_node->args) {
        for (auto n : VisitExpr(arg)) {
          inputs.push_back(n);
        }
      }
    } else {
      LOG(FATAL) << "Non-primitive-call nodes should have been transformed away.\n"
                 << "The graph executor code generator expects all calls to be call_lowered, "
                 << "but found: " << std::endl
                 << PrettyPrint(call);
    }

    // Compute the operator name, because we used the get unique name when generating the kernel.
    auto op_name = name_supply_->FreshName(func_name);
    auto node = GraphOpNode::make_node_ptr(op_name, GraphAttrs(), func_name, inputs, attrs);
    return AddNode(node, call);
  }

  std::vector<GraphNodeRef> VisitExpr_(const CallNode* call_node) override {
    relay::Call call = GetRef<Call>(call_node);
    OnDeviceProps props = GetOnDeviceProps(call_node);
    if (props.body.defined()) {
      // See through "on_device" calls.
      return VisitExpr(props.body);
    }
    return GraphAddCallNode(call_node, GraphAttrs());
  }

  std::vector<GraphNodeRef> VisitExpr_(const LetNode* op) override {
    ICHECK_EQ(var_map_.count(op->var.get()), 0);
    var_map_[op->var.get()] = VisitExpr(op->value);
    return VisitExpr(op->body);
  }
  std::vector<GraphNodeRef> VisitExpr_(const TupleGetItemNode* op) override {
    auto vtuple = VisitExpr(op->tuple);
    return {vtuple[op->index]};
  }

  std::vector<GraphNodeRef> VisitExpr_(const OpNode* op) override {
    LOG(FATAL) << "All OpNodes should have been expanded";
  }
  std::vector<GraphNodeRef> VisitExpr_(const GlobalVarNode* op) override {
    LOG(FATAL) << "All GlobalVarNodes should be removed before graph executor's Codegen is called";
  }
  std::vector<GraphNodeRef> VisitExpr_(const IfNode* op) override {
    LOG(FATAL) << "Graph executor does not support control flow (found IfNode)";
  }
  std::vector<GraphNodeRef> VisitExpr_(const FunctionNode* op) override {
    ICHECK(op->GetAttr<String>(attr::kCompiler).defined())
        << "Only functions supported by custom codegen";
    return {};
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefCreateNode* op) override {
    LOG(FATAL) << "Graph executor does not support references (found RefCreateNode)";
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefReadNode* op) override {
    LOG(FATAL) << "Graph executor does not support references (found RefReadNode)";
  }
  std::vector<GraphNodeRef> VisitExpr_(const RefWriteNode* op) override {
    LOG(FATAL) << "Graph executor does not support references (found RefWriteNode)";
  }
  std::vector<GraphNodeRef> VisitExpr_(const ConstructorNode* op) override {
    LOG(FATAL) << "Graph executor does not support ADTs (found ConstructorNode)";
  }
  std::vector<GraphNodeRef> VisitExpr_(const MatchNode* op) override {
    LOG(FATAL) << "Graph executor does not support matching (found MatchNode)";
  }
  /*!
   * \brief Generate Graph JSON
   *
   * \param writer json writer
   */
  void GetJSON(dmlc::JSONWriter* writer) {
    std::vector<size_t> arg_nodes;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      auto node = nodes_[i];
      if (node->Type() == kGraphInputNode) {
        arg_nodes.push_back(i);
      }
    }
    size_t num_entry = 0;
    ShapeVector shapes;
    std::vector<size_t> storage_ids;
    std::vector<std::string> storage_scopes;
    std::vector<size_t> device_types;
    std::vector<std::string> dltypes;
    std::vector<size_t> node_row_ptr{0};
    for (auto node : nodes_) {
      const auto& shape_vec = dmlc::get<ShapeVector>(node->attrs_["shape"]);
      const auto& storage_id = dmlc::get<std::vector<int64_t>>(node->attrs_["storage_id"]);
      const auto& storage_scope =
          dmlc::get<std::vector<std::string>>(node->attrs_["storage_scope"]);
      const auto& dtype_vec = dmlc::get<std::vector<std::string>>(node->attrs_["dtype"]);

      ICHECK_EQ(node->num_outputs_, shape_vec.size());
      num_entry += node->num_outputs_;

      shapes.insert(shapes.end(), shape_vec.begin(), shape_vec.end());
      dltypes.insert(dltypes.end(), dtype_vec.begin(), dtype_vec.end());
      storage_ids.insert(storage_ids.end(), storage_id.begin(), storage_id.end());
      storage_scopes.insert(storage_scopes.end(), storage_scope.begin(), storage_scope.end());
      if (node->attrs_.count("device_index")) {
        const auto& dev_types = dmlc::get<std::vector<int64_t>>(node->attrs_["device_index"]);
        device_types.insert(device_types.end(), dev_types.begin(), dev_types.end());
      }
      node_row_ptr.push_back(num_entry);
    }

    // verification if storage_scope contains any non global memory scope
    // in other case it's better not to write scopes to the JSON at all
    bool global_only_scope = true;
    for (const auto& ss : storage_scopes) {
      if (!(ss.empty() || ss == "global")) {
        global_only_scope = false;
      }
    }
    if (global_only_scope) {
      storage_scopes.clear();
    }
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes_);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
    writer->WriteObjectKeyValue("heads", heads_);
    std::unordered_map<std::string, std::vector<dmlc::any>> attrs;
    attrs["shape"].emplace_back(std::string("list_shape"));
    attrs["shape"].emplace_back(shapes);
    attrs["storage_id"].emplace_back(std::string("list_int"));
    attrs["storage_id"].emplace_back(storage_ids);
    if (device_types.size()) {
      attrs["device_index"].emplace_back(std::string("list_int"));
      attrs["device_index"].emplace_back(device_types);
    }
    if (storage_scopes.size()) {
      attrs["storage_scope"].emplace_back(std::string("list_str"));
      attrs["storage_scope"].emplace_back(storage_scopes);
    }
    attrs["dltype"].emplace_back(std::string("list_str"));
    attrs["dltype"].emplace_back(dltypes);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->EndObject();
  }

 protected:
  /*! \brief nodes */
  std::vector<GraphObjectPtr> nodes_;
  /*! \brief output of graph */
  std::vector<GraphNodeRef> heads_;
  /*! \brief mod */
  runtime::Module* mod_;
  /*! \brief variable map */
  std::unordered_map<const Object*, std::vector<GraphNodeRef>> var_map_;
  /*! \brief Available targets */
  CompilationConfig config_;
  /*!
   * \brief parameters (i.e. ConstantNodes found in the graph).
   * These are take as inputs to the GraphExecutor.
   * Maps param name to a pair of storage_id and NDArray. At runtime, the storage_id can be
   * used to lookup the parameter.
   */
  std::unordered_map<std::string, runtime::NDArray> params_;
  std::unordered_map<std::string, int64_t> param_storage_ids_;
  /*! \brief plan memory of device result */
  StaticMemoryPlan memory_plan_;
  /*! \brief the module name we use to mangle the function names */
  String mod_name_;
  /*! \brief function metadata */
  Map<String, FunctionInfo> function_metadata_;
  /*! \brief NameSupply */
  NameSupply name_supply_ = NameSupply("");
};

class GraphExecutorCodegenModule : public runtime::ModuleNode {
 public:
  GraphExecutorCodegenModule() {}
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2) << "The expected of arguments are: "
                                    << "runtime::Module mod and Array<Target> targets";
        void* mod = args[0];
        Array<Target> targets = args[1];
        codegen_ = std::make_shared<GraphExecutorCodegen>(reinterpret_cast<runtime::Module*>(mod),
                                                          std::move(targets));
      });
    } else if (name == "codegen") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        IRModule mod = args[0];
        Function func = args[1];
        String mod_name = args[2];
        this->output_ = this->codegen_->Codegen(mod, func, mod_name);
      });
    } else if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->output_.graph_json; });
    } else if (name == "list_params_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Array<runtime::String> ret;
        for (const auto& kv : this->output_.params) {
          ret.push_back(kv.first);
        }
        *rv = ret;
      });
    } else if (name == "get_param_by_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        String key = args[0];
        auto it = this->output_.params.find(key);
        CHECK(it != this->output_.params.end()) << "no such parameter " << key;
        *rv = (*it).second;
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.lowered_funcs;
      });
    } else if (name == "get_external_modules") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.external_mods;
      });
    } else if (name == "get_devices") {
      return PackedFunc([sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = Array<String>(); });
    } else if (name == "get_executor_codegen_metadata") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->output_.metadata; });
    } else if (name == "get_function_metadata") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.function_metadata;
      });
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  const char* type_key() const final { return "RelayGraphExecutorCodegenModule"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return runtime::ModulePropertyMask::kRunnable; }

 private:
  std::shared_ptr<GraphExecutorCodegen> codegen_;
  LoweredOutput output_;
};

runtime::Module CreateGraphCodegenMod() {
  auto ptr = make_object<GraphExecutorCodegenModule>();
  return runtime::Module(ptr);
}

TVM_REGISTER_GLOBAL("relay.build_module._GraphExecutorCodegen")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = CreateGraphCodegenMod(); });

}  // namespace backend
}  // namespace relay
}  // namespace tvm

namespace dmlc {
namespace json {
// JSON utils
template <typename T>
inline bool SameType(const dmlc::any& data) {
  return std::type_index(data.type()) == std::type_index(typeid(T));
}

template <>
struct Handler<std::shared_ptr<tvm::relay::backend::GraphNode>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::shared_ptr<tvm::relay::backend::GraphNode>& data) {
    data->Save(writer);
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::shared_ptr<tvm::relay::backend::GraphNode>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};
template <>
struct Handler<std::unordered_map<std::string, dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::unordered_map<std::string, dmlc::any>& data) {
    writer->BeginObject();
    for (const auto& kv : data) {
      auto k = kv.first;
      const dmlc::any& v = kv.second;
      if (SameType<std::string>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::string>>(v));
      } else if (SameType<std::vector<dmlc::any>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<dmlc::any>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndObject();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::unordered_map<std::string, dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

template <>
struct Handler<std::vector<dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer, const std::vector<dmlc::any>& data) {
    writer->BeginArray();
    for (const auto& v : data) {
      if (SameType<std::string>(v)) {
        writer->WriteArrayItem(dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteArrayItem(dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::string>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, std::vector<dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};
}  // namespace json
}  // namespace dmlc
