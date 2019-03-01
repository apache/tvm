/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/compile_engine.h
 * \brief Internal compilation engine handle function cache.
 *  and interface to low level code generation.
 */
#ifndef TVM_RELAY_FRONTEND_GRAPH_RUNTIME_CODEGEN_H_
#define TVM_RELAY_FRONTEND_GRAPH_RUNTIME_CODEGEN_H_

#include <dmlc/any.h>
#include <dmlc/json.h>
#include <tvm/node/ir_functor.h>
#include <tvm/relay/expr_functor.h>

#include <list>
#include <string>
#include <vector>

#include "../backend/compile_engine.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace frontend {

class _Node;
class _InputNode;
class _OpNode;

using IntegerArray = Array<Integer>;
using ShapeVector = std::vector<std::vector<int64_t>>;
using Attrs = std::unordered_map<std::string, dmlc::any>;
using _NodePtr = std::shared_ptr<_Node>;
using _InputNodePtr = std::shared_ptr<_InputNode>;
using _OpNodePtr = std::shared_ptr<_OpNode>;

/*! \brief Lowered outputs */
struct LoweredOutput {
  std::string graph_json;
  std::unordered_map<std::string, std::vector<LoweredFunc>> lowered_funcs;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*! \brief Node types */
enum _NodeType {
  kNop_,
  kInputNode_,
  kOpNode_,
};

/*! \brief Node Reference */
class _NodeRef {
 public:
  _NodeRef() {}

  _NodeRef(int ident, int index, int version = 0)
      : ident_(ident), index_(index), version_(version) {}

  inline void Save(dmlc::JSONWriter* writer) const {
    writer->BeginArray();
    writer->WriteArrayItem(ident_);
    writer->WriteArrayItem(index_);
    writer->WriteArrayItem(version_);
    writer->EndArray();
  }

  inline void Load(dmlc::JSONReader* reader) {}

 protected:
  int ident_;
  int index_{0};
  int version_{0};
};

/*! \brief Base Node class */
class _Node {
 public:
  _Node() {}
  virtual void Save(dmlc::JSONWriter* writer) const {}
  virtual void Load(dmlc::JSONReader* reader) {}
  virtual _NodeType Type() const { return kNop_; }
  virtual ~_Node() {}

 public:
  int num_outputs_{1};
  std::string name_;
  std::unordered_map<std::string, dmlc::any> attrs_;
};

/*! \brief Input Node */
class _InputNode : public _Node {
 public:
  _InputNode() {}
  _InputNode(const std::string& name, const Attrs& attrs) {
    name_ = name;
    attrs_ = attrs;
  }

  _NodeType Type() const override { return kInputNode_; }
  void Load(dmlc::JSONReader* reader) override {}
  void Save(dmlc::JSONWriter* writer) const override {
    const std::string op_name{"null"};
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_name);
    writer->WriteObjectKeyValue("name", this->name_);
    writer->WriteObjectKeyValue("inputs", std::list<int>());
    writer->EndObject();
  }
  static std::shared_ptr<_Node> make_node_ptr(const std::string& name, const Attrs& attrs) {
    auto ptr = std::make_shared<_InputNode>(name, attrs);
    return std::dynamic_pointer_cast<_Node>(ptr);
  }
};

/*! \brief Op Node */
class _OpNode : public _Node {
 public:
  _OpNode() {}
  _OpNode(const std::string& name, const Attrs& nd_attrs, const std::string& op_name,
          const std::vector<_NodeRef>& inputs, const Attrs& attrs, size_t num_outputs = 1) {
    name_ = name;
    attrs_ = nd_attrs;
    op_name_ = op_name;
    inputs_ = inputs;
    op_attrs_ = attrs_;
    num_outputs_ = num_outputs;
    op_attrs_["func_name"] = op_name_;
    op_attrs_["flatten_data"] = std::string("0");
    op_attrs_["num_inputs"] = std::to_string(inputs_.size());
    op_attrs_["num_outputs"] = std::to_string(num_outputs_);
  }

  _NodeType Type() const override { return kOpNode_; }
  void Load(dmlc::JSONReader* reader) override {}
  void Save(dmlc::JSONWriter* writer) const override {
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_type_name_);
    writer->WriteObjectKeyValue("name", name_);
    writer->WriteObjectKeyValue("attrs", op_attrs_);
    writer->WriteObjectKeyValue("inputs", this->inputs_);
    writer->EndObject();
  }
  static std::shared_ptr<_Node> make_node_ptr(const std::string& name, const Attrs& nd_attrs,
                                              const std::string& op_name,
                                              const std::vector<_NodeRef>& inputs,
                                              const Attrs& attrs, size_t num_outputs = 1) {
    auto ptr = std::make_shared<_OpNode>(name, nd_attrs, op_name, inputs, attrs, num_outputs);
    return std::dynamic_pointer_cast<_Node>(ptr);
  }

 public:
  std::string op_name_;
  std::vector<_NodeRef> inputs_;
  Attrs op_attrs_;

 private:
  const std::string op_type_name_{"tvm_op"};
};

/*! \brief Code generator for graph runtime */
class GraphRuntimeCodegen
    : public ::tvm::relay::ExprFunctor<std::vector<_NodeRef>(const Expr&, const Expr&)> {
 public:
  GraphRuntimeCodegen(runtime::Module* mod, tvm::Target target) : mod_(mod), target_(target) {
    compile_engine_ = CompileEngine::Global();
  }

  LoweredOutput Codegen(relay::Function func) {
    auto pf = GetPakcedFunc("relay.backend.GraphPlanMemory");
    storage_device_map_ = (*pf)(func);
    // First we convert all the parameters into input nodes.
    for (auto param : func->params) {
      auto node_ptr = _InputNode::make_node_ptr(param->name_hint(), Attrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }
    heads_ = VisitExpr(func->body, func->body);
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    LoweredOutput ret;
    ret.graph_json = os.str();
    ret.params = params_;
    for (auto& kv : lowered_funcs_) {
      auto& vec = ret.lowered_funcs[kv.first];
      vec.insert(vec.begin(), kv.second.begin(), kv.second.end());
    }
    return ret;
  }

 protected:
  /*!
   * \brief Extract shape from expr to vector<int64_t>
   *
   * \param shape
   * \return std::vector<int64_t>
   */
  std::vector<int64_t> _ShapeToJSON(tvm::Array<HalideIR::Expr> shape) {
    std::vector<int64_t> ret;
    for (IndexExpr dim : shape) {
      const int64_t* pval = as_const_int(dim);
      ret.push_back(*pval);
    }
    return ret;
  }

  /*!
   * \brief Add node to graph
   *
   * \param node
   * \param expr
   * \return std::vector<_NodeRef>
   */
  std::vector<_NodeRef> AddNode(_NodePtr node, Expr expr) {
    auto checked_type = expr->checked_type();
    size_t count = storage_device_map_.count(expr);
    CHECK_GT(count, 0) << "Expr is not existing in storage plan";
    auto storage_device_info = storage_device_map_[expr];
    CHECK_EQ(storage_device_info.size(), 2);
    // storage
    std::vector<int64_t> storage_info;
    for (auto& v : storage_device_info[0]) {
      storage_info.push_back(v->value);
    }
    node->attrs_["storage_id"] = std::move(storage_info);
    // type
    std::vector<int64_t> device_types;
    for (auto& v : storage_device_info[1]) {
      device_types.push_back(v->value);
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
    auto node_id = nodes_.size();
    nodes_.push_back(node);
    // Tuple return value, flatten as tuple
    if (const auto* tuple_type = checked_type.as<TupleTypeNode>()) {
      std::vector<_NodeRef> ret;
      ShapeVector shape;
      std::vector<std::string> dtype;
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        if (const auto* typ = tuple_type->fields[i].as<TensorTypeNode>()) {
          ret.push_back(_NodeRef(node_id, i));
          shape.emplace_back(_ShapeToJSON(typ->shape));
          dtype.emplace_back(DType2String(typ->dtype));
        } else {
          LOG(FATAL) << "type " << checked_type->type_key() << " not supported";
        }
      }
      CHECK_EQ(node->Type(), kOpNode_);
      auto op_nd = std::dynamic_pointer_cast<_OpNode>(node);
      op_nd->attrs_["shape"] = shape;
      op_nd->attrs_["dtype"] = dtype;
      op_nd->num_outputs_ = tuple_type->fields.size();
      return ret;
    }
    // Normal tensor return type
    if (const auto* tensor_type = checked_type.as<TensorTypeNode>()) {
      ShapeVector shape;
      std::vector<std::string> dtype;
      shape.emplace_back(_ShapeToJSON(tensor_type->shape));
      dtype.emplace_back(DType2String(tensor_type->dtype));
      node->attrs_["shape"] = shape;
      node->attrs_["dtype"] = dtype;
    } else {
      LOG(FATAL) << "type " << checked_type->type_key() << " not supported";
    }
    return {_NodeRef(node_id, 0)};
  }

  /*! \brief Visitors */
  std::unordered_map<Expr, std::vector<_NodeRef>, NodeHash, NodeEqual> visitor_cache_;

  std::vector<_NodeRef> VisitExpr(const Expr& expr, const Expr& dummy) override {
    if (visitor_cache_.count(expr)) return visitor_cache_.at(expr);
    std::vector<_NodeRef> res;
    if (expr.as<ConstantNode>()) {
      res = VisitExpr_(expr.as<ConstantNode>(), expr);
    } else if (expr.as<TupleNode>()) {
      res = VisitExpr_(expr.as<TupleNode>(), expr);
    } else if (expr.as<VarNode>()) {
      res = VisitExpr_(expr.as<VarNode>(), expr);
    } else if (expr.as<GlobalVarNode>()) {
      res = VisitExpr_(expr.as<GlobalVarNode>(), expr);
    } else if (expr.as<FunctionNode>()) {
      res = VisitExpr_(expr.as<FunctionNode>(), expr);
    } else if (expr.as<CallNode>()) {
      res = VisitExpr_(expr.as<CallNode>(), expr);
    } else if (expr.as<LetNode>()) {
      res = VisitExpr_(expr.as<LetNode>(), expr);
    } else if (expr.as<IfNode>()) {
      res = VisitExpr_(expr.as<IfNode>(), expr);
    } else if (expr.as<OpNode>()) {
      res = VisitExpr_(expr.as<OpNode>(), expr);
    } else if (expr.as<TupleGetItemNode>()) {
      res = VisitExpr_(expr.as<TupleGetItemNode>(), expr);
    } else if (expr.as<RefCreateNode>()) {
      res = VisitExpr_(expr.as<RefCreateNode>(), expr);
    } else if (expr.as<RefReadNode>()) {
      res = VisitExpr_(expr.as<RefReadNode>(), expr);
    } else if (expr.as<RefWriteNode>()) {
      res = VisitExpr_(expr.as<RefWriteNode>(), expr);
    } else if (expr.as<ConstructorNode>()) {
      res = VisitExpr_(expr.as<ConstructorNode>(), expr);
    } else if (expr.as<MatchNode>()) {
      res = VisitExpr_(expr.as<MatchNode>(), expr);
    }
    return res;
  }

  std::vector<_NodeRef> VisitExpr_(const VarNode* op, const Expr& expr) override {
    return var_map_[expr.get()];
  }

  std::vector<_NodeRef> VisitExpr_(const ConstantNode* op, const Expr& expr) override {
    size_t index = params_.size();
    std::string name = "p" + std::to_string(index);
    params_[name] = op->data;
    auto node = _InputNode::make_node_ptr(name, Attrs());
    return AddNode(node, expr);
  }

  std::vector<_NodeRef> VisitExpr_(const TupleNode* op, const Expr& expr) override {
    std::vector<_NodeRef> fields;
    for (auto field : op->fields) {
      auto ref_vec = VisitExpr(field, field);
      for (auto ref : ref_vec) {
        fields.push_back(ref);
      }
    }
    return fields;
  }
  std::vector<_NodeRef> VisitExpr_(const CallNode* op, const Expr& expr) override {
    Expr func;
    if (op->op.as<OpNode>()) {
      LOG(FATAL) << "Operators should be transformed away; try applying"
                 << "the fuse_ops transformation to the expression.";
    } else if (op->op.as<GlobalVarNode>()) {
      LOG(FATAL) << "Not implemented";
    } else if (op->op.as<FunctionNode>()) {
      func = op->op;
    } else {
      LOG(FATAL) << "TVM runtime does not support calls to " << op->op->type_key();
    }
    /*!
    if int(func.attrs.Primitive) != 1:
            raise Exception(
                "TVM only support calls to primitive functions " +
                "(i.e functions composed of fusable operator invocations)")
    */
    // Only support homogeneous execution for now
    CHECK_GE(storage_device_map_.count(expr), 0);
    // TODO(xxx): heterogeneous
    // auto &device_type = storage_device_map_[expr][1];
    // auto call_dev_type = device_type[0]->value;
    auto pf0 = GetPakcedFunc("relay.backend._make_CCacheKey");
    CCacheKey key = (*pf0)(func, target_);
    auto pf1 = GetPakcedFunc("relay.backend._CompileEngineLower");
    CachedFunc lowerd_func = (*pf1)(compile_engine_, key);
    if (lowered_funcs_.count(target_->target_name)) {
      lowered_funcs_[target_->target_name] = {};
    }
    for (auto f : lowerd_func->funcs) {
      lowered_funcs_[target_->target_name].insert(f);
    }
    std::vector<_NodeRef> inputs;
    for (auto arg : op->args) {
      auto res = VisitExpr(arg, arg);
      for (auto nr : res) {
        inputs.push_back(nr);
      }
    }
    auto& op_name = lowerd_func->func_name;
    auto node = _OpNode::make_node_ptr(_GetUniqueName(op_name), Attrs(), op_name, inputs, Attrs());
    return AddNode(node, expr);
  }

  std::vector<_NodeRef> VisitExpr_(const LetNode* op, const Expr& expr) override {
    CHECK_EQ(var_map_.count(op->var.get()), 0);
    var_map_[op->var.get()] = VisitExpr(op->value, op->value);
    return VisitExpr(op->body, op->body);
  }
  std::vector<_NodeRef> VisitExpr_(const TupleGetItemNode* op, const Expr& expr) override {
    auto vtuple = VisitExpr(op->tuple, op->tuple);
    return {vtuple[op->index]};
  }
  std::vector<_NodeRef> VisitExpr_(const OpNode* op, const Expr& expr) override {
    throw std::runtime_error("can not compile op in non-eta expanded form");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const GlobalVarNode* op, const Expr& expr) override {
    throw std::runtime_error("");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const IfNode* op, const Expr& expr) override {
    throw std::invalid_argument("if not supported");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const FunctionNode* op, const Expr& expr) override {
    throw std::invalid_argument("function not supported");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const RefCreateNode* op, const Expr& expr) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const RefReadNode* op, const Expr& expr) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const RefWriteNode* op, const Expr& expr) override {
    throw std::invalid_argument("reference not supported");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const ConstructorNode* op, const Expr& expr) override {
    throw std::invalid_argument("ADT constructor case not yet implemented");
    return {};
  }
  std::vector<_NodeRef> VisitExpr_(const MatchNode* op, const Expr& expr) override {
    throw std::invalid_argument("match case not yet implemented");
    return {};
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
      if (node->Type() == kInputNode_) {
        arg_nodes.push_back(i);
      }
    }
    size_t num_entry = 0;
    ShapeVector shapes;
    std::vector<size_t> storage_ids;
    std::vector<size_t> device_types;
    std::vector<std::string> dltypes;
    std::vector<size_t> node_row_ptr{0};
    for (auto node : nodes_) {
      const auto& shape_vec = dmlc::get<ShapeVector>(node->attrs_["shape"]);
      const auto& storage_id = dmlc::get<std::vector<int64_t>>(node->attrs_["storage_id"]);
      const auto& dtype_vec = dmlc::get<std::vector<std::string>>(node->attrs_["dtype"]);

      CHECK_EQ(node->num_outputs_, shape_vec.size());
      num_entry += node->num_outputs_;

      shapes.insert(shapes.end(), shape_vec.begin(), shape_vec.end());
      dltypes.insert(dltypes.end(), dtype_vec.begin(), dtype_vec.end());
      storage_ids.insert(storage_ids.end(), storage_id.begin(), storage_id.end());
      if (node->attrs_.count("device_index")) {
        const auto& dev_types = dmlc::get<std::vector<int64_t>>(node->attrs_["device_index"]);
        device_types.insert(device_types.end(), dev_types.begin(), dev_types.end());
      }
      node_row_ptr.push_back(num_entry);
    }
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes_);
    // WriteNodePtrVecJSON("nodes", nodes_, writer);
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
    attrs["dltype"].emplace_back(std::string("list_str"));
    attrs["dltype"].emplace_back(dltypes);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->EndObject();
  }

  /*!
   * \brief Get unique name for func
   *
   * \param name
   * \return std::string
   */
  std::string _GetUniqueName(const std::string& name) {
    if (!name_map_.count(name)) {
      name_map_[name] = 1;
      return name;
    }
    auto index = name_map_[name];
    name_map_[name] += 1;
    return _GetUniqueName(name + std::to_string(index));
  }

 protected:
  /*! \brief nodes */
  std::vector<_NodePtr> nodes_;
  /*! \brief output of graph */
  std::vector<_NodeRef> heads_;
  /*! \brief ? */
  runtime::Module* mod_;
  /*! \brief variable map */
  std::unordered_map<const Node*, std::vector<_NodeRef>> var_map_;
  /*! \brief target device */
  tvm::Target target_;
  /*! \brief params */
  std::unordered_map<std::string, runtime::NDArray> params_;
  /*! \brief plan memory of device result */
  Map<Expr, Array<IntegerArray>> storage_device_map_;
  /*! \brief lowered funcs */
  std::unordered_map<std::string, std::unordered_set<LoweredFunc, NodeHash, NodeEqual>>
      lowered_funcs_;
  /*! \brief name map */
  std::unordered_map<std::string, size_t> name_map_;
  /*! \brief compile engine */
  CompileEngine compile_engine_;
};

}  // namespace frontend
}  // namespace relay
}  // namespace tvm

namespace dmlc {
namespace json {

template <typename T>
inline bool SameType(const dmlc::any& data) {
  return std::type_index(data.type()) == std::type_index(typeid(T));
}

template <>
struct Handler<std::shared_ptr<tvm::relay::frontend::_Node>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::shared_ptr<tvm::relay::frontend::_Node>& data) {
    data->Save(writer);
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::shared_ptr<tvm::relay::frontend::_Node>* data) {
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

#endif  // TVM_RELAY_FRONTEND_GRAPH_RUNTIME_CODEGEN_H_
