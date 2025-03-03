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
 * \file relax/backend/contrib/codegen_json/codegen_json.h
 * \brief Utilities for json codegen and runtime
 */
#ifndef TVM_RELAX_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_
#define TVM_RELAX_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_

#include <dmlc/any.h>
#include <dmlc/json.h>
#include <tvm/node/reflection.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../../../../runtime/contrib/json/json_runtime.h"
#include "../../../transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace backend {
namespace contrib {

using namespace tvm::runtime::json;

using ShapeVector = std::vector<std::vector<int64_t>>;
using TypeVector = std::vector<std::string>;
using JSONGraphObjectPtr = std::shared_ptr<JSONGraphNode>;

/*!
 * \brief Helper class to extract all attributes of a certain op and save them
 * into text format.
 */
class OpAttrExtractor : public AttrVisitor {
 public:
  explicit OpAttrExtractor(JSONGraphObjectPtr node) : node_(node) {}

  template <typename T = double, typename = std::enable_if_t<std::is_floating_point<T>::value>>
  std::string Fp2String(const T value) {
    std::ostringstream out;
    out.precision(std::numeric_limits<T>::max_digits10);
    out << value;
    return out.str();
  }

  void SetNodeAttr(const char* key, const std::vector<std::string>& value) {
    std::vector<dmlc::any> attr;
    attr.emplace_back(value);
    node_->SetAttr(key, attr);
  }

  void Visit(const char* key, double* value) final { SetNodeAttr(key, {Fp2String(*value)}); }

  void Visit(const char* key, int64_t* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, uint64_t* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, int* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, bool* value) final { SetNodeAttr(key, {std::to_string(*value)}); }

  void Visit(const char* key, std::string* value) final { SetNodeAttr(key, {*value}); }

  void Visit(const char* key, DataType* value) final {
    if (!value->is_void()) {
      SetNodeAttr(key, {runtime::DLDataType2String(*value)});
    } else {
      SetNodeAttr(key, {""});
    }
  }

  void Visit(const char* key, runtime::ObjectRef* value) final {
    if (const auto* an = (*value).as<ArrayNode>()) {
      std::vector<std::string> attr;
      for (size_t i = 0; i < an->size(); ++i) {
        if (const auto* im = (*an)[i].as<IntImmNode>()) {
          attr.push_back(std::to_string(im->value));
        } else if (const auto* fm = (*an)[i].as<FloatImmNode>()) {
          attr.push_back(Fp2String(fm->value));
        } else if (const auto* str = (*an)[i].as<StringObj>()) {
          String s = GetRef<String>(str);
          attr.push_back(s);
        } else {
          LOG(FATAL) << "Not supported type: " << (*an)[i]->GetTypeKey();
        }
      }
      SetNodeAttr(key, attr);
    } else if (!(*value).defined()) {  // Skip NullValue
      SetNodeAttr(key, std::vector<std::string>{""});
    } else if (const auto* im = (*value).as<IntImmNode>()) {
      SetNodeAttr(key, std::vector<std::string>{std::to_string(im->value)});
    } else if (const auto* fm = (*value).as<FloatImmNode>()) {
      SetNodeAttr(key, std::vector<std::string>{Fp2String(fm->value)});
    } else if (const auto* str = (*value).as<StringObj>()) {
      String s = GetRef<String>(str);
      SetNodeAttr(key, std::vector<std::string>{s});
    } else {
      LOG(FATAL) << "Not yet supported type: " << (*value)->GetTypeKey() << ": " << *value;
    }
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "NDArray is not allowed in op attribute";
  }

  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "void pointer is not allowed in op attribute";
  }

  void Extract(Object* node) {
    if (node) {
      reflection_->VisitAttrs(node, this);
    }
  }

 private:
  JSONGraphObjectPtr node_;
  ReflectionVTable* reflection_ = ReflectionVTable::Global();
};

using NodeEntries = std::vector<JSONGraphNodeEntry>;

/*! \brief Serialize a Relax expression to JSON. */
class JSONSerializer : public relax::MemoizedExprTranslator<NodeEntries> {
 public:
  using MemoizedExprTranslator<NodeEntries>::VisitExpr_;
  using MemoizedExprTranslator<NodeEntries>::VisitBinding_;

  /*!
   * \brief Constructor
   * \param constant_names The names of all constants in the original module.
   */
  explicit JSONSerializer(const Map<Constant, String>& constant_names)
      : constant_names_(constant_names) {}

  void serialize(Function func) {
    // First we convert all the parameters into input nodes.
    for (const auto& param : func->params) {
      auto node_ptr = std::make_shared<JSONGraphNode>(param->name_hint(), "input" /* op_type_ */);
      memo_[param] = AddNode(node_ptr, param);
    }
    heads_ = VisitExpr(func->body);
  }

  /*!\brief Return the required constants. */
  Array<String> GetConstantNames() const { return constants_used_; }

  /*!\brief Return the generated json. */
  std::string GetJSON() {
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    Save(&writer);
    return os.str();
  }

 protected:
  /*!
   * \brief Add a node to graph.
   *
   * \param node A graph node. It is a shared pointer. Some attributes of it
   *        will be added, i.e. shape and type. These attributes are attached to
   *        the JSON graph in the end.
   * \param expr The relax expression.
   * \return A list of graph entry nodes. It the relax expr is a tuple type, we
   *         will flatten it.
   */
  NodeEntries AddNode(JSONGraphObjectPtr node, const Expr& expr) {
    auto struct_info = GetStructInfo(expr);
    auto node_id = nodes_.size();
    nodes_.push_back(node);
    NodeEntries ret;
    ShapeVector shape;
    TypeVector dtype;

    // Flatten tuple node.
    if (const auto* tuple_sinfo = struct_info.as<TupleStructInfoNode>()) {
      for (size_t i = 0; i < tuple_sinfo->fields.size(); ++i) {
        const auto* tensor_sinfo = tuple_sinfo->fields[i].as<TensorStructInfoNode>();
        ICHECK(tensor_sinfo) << "Expect TensorStructInfo, but received: ."
                             << tuple_sinfo->fields[i]->GetTypeKey();
        ICHECK(tensor_sinfo->shape.defined()) << "Expect shape to be defined.";
        ShapeExpr output_shape = Downcast<ShapeExpr>(tensor_sinfo->shape.value());
        ret.push_back(JSONGraphNodeEntry(node_id, i));
        shape.emplace_back(GetIntShape(output_shape->values));
        dtype.emplace_back(DType2String(tensor_sinfo->dtype));
      }
      node->SetNumOutput(tuple_sinfo->fields.size());
    } else {
      const auto* tensor_sinfo = struct_info.as<TensorStructInfoNode>();
      ICHECK(tensor_sinfo) << "Expect TensorStructInfo, but received: "
                           << struct_info->GetTypeKey();
      ICHECK(tensor_sinfo->shape.defined()) << "Expect shape to be defined.";
      ShapeExpr output_shape = Downcast<ShapeExpr>(tensor_sinfo->shape.value());

      shape.emplace_back(GetIntShape(output_shape->values));
      dtype.emplace_back(DType2String(tensor_sinfo->dtype));
      ret.push_back(JSONGraphNodeEntry(node_id, 0));
    }
    std::vector<dmlc::any> shape_attrs;
    shape_attrs.emplace_back(shape);
    node->SetAttr("shape", shape_attrs);

    std::vector<dmlc::any> type_attrs;
    type_attrs.emplace_back(dtype);
    node->SetAttr("dtype", type_attrs);
    return ret;
  }

  void SetCallNodeAttribute(JSONGraphObjectPtr node, const CallNode* cn) {
    if (cn->op.as<OpNode>()) {
      OpAttrExtractor extractor(node);
      const Object* call_attr = cn->attrs.get();
      extractor.Extract(const_cast<Object*>(call_attr));
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      ICHECK(false);
      auto pattern = fn->GetAttr<String>(attr::kPartitionedFromPattern);
      ICHECK(pattern.defined());
      std::vector<std::string> values;
      values.push_back(pattern.value());
      std::vector<dmlc::any> attr;
      attr.emplace_back(values);
      node->SetAttr("PartitionedFromPattern", attr);
    }
  }

  NodeEntries VisitBinding_(const MatchCastNode* binding) {
    LOG(FATAL) << "JSON runtime currently doesn't match cast\n";
    return {};
  }

  NodeEntries VisitBinding(const Binding& binding) {
    NodeEntries nodes;
    if (const auto* node = binding.as<VarBindingNode>()) {
      auto from_b = VisitBinding_(node);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    } else if (const auto* node = binding.as<MatchCastNode>()) {
      auto from_b = VisitBinding_(node);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
    }
    return nodes;
  }

  NodeEntries VisitBindingBlock(const BindingBlock& block) {
    NodeEntries nodes;
    if (const auto* node = block.as<DataflowBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      nodes.insert(nodes.end(), from_bb.begin(), from_bb.end());
    } else if (const auto* node = block.as<BindingBlockNode>()) {
      auto from_bb = VisitBindingBlock_(node);
      nodes.insert(nodes.end(), from_bb.begin(), from_bb.end());
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
    }
    return nodes;
  }

  NodeEntries VisitBindingBlock_(const BindingBlockNode* block) {
    NodeEntries nodes;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    }
    return nodes;
  }

  NodeEntries VisitBindingBlock_(const DataflowBlockNode* block) {
    NodeEntries nodes;
    for (Binding binding : block->bindings) {
      auto from_b = VisitBinding(binding);
      nodes.insert(nodes.end(), from_b.begin(), from_b.end());
    }
    return nodes;
  }

  NodeEntries VisitExpr_(const SeqExprNode* op) {
    NodeEntries nodes;
    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }
    auto from_body = VisitExpr(op->body);
    nodes.insert(nodes.end(), from_body.begin(), from_body.end());
    return nodes;
  }

  NodeEntries VisitExprDefault_(const Object* op) {
    LOG(FATAL) << "JSON runtime currently doesn't support " << op->GetTypeKey();
    return {};
  }

  NodeEntries VisitExpr_(const ConstantNode* cn) {
    auto name = constant_names_.find(GetRef<Constant>(cn));
    ICHECK(name != constant_names_.end())
        << "Cannot find the name of the constant: " << GetRef<Constant>(cn);
    constants_used_.push_back((*name).second);
    auto node = std::make_shared<JSONGraphNode>((*name).second, "const" /* op_type_ */);
    return AddNode(node, GetRef<Expr>(cn));
  }

  NodeEntries VisitExpr_(const TupleNode* tn) {
    NodeEntries fields;
    for (const auto& field : tn->fields) {
      auto ref = VisitExpr(field);
      fields.insert(fields.end(), ref.begin(), ref.end());
    }
    return fields;
  }

  NodeEntries VisitExpr_(const CallNode* cn) {
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      ICHECK(comp.defined()) << "JSON runtime only supports composite functions.";
      name = comp.value();
    } else {
      LOG(FATAL) << "JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    // TODO(@sunggg): Revisit when we have op naming convention.
    // Currently, simply remove "relax." prefix to make it work.
    name = std::string("jsonruntime.") + name.substr(6);

    NodeEntries inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    SetCallNodeAttribute(node, cn);
    return AddNode(node, GetRef<Expr>(cn));
  }

  NodeEntries VisitExpr_(const TupleGetItemNode* gtn) {
    auto vtuple = VisitExpr(gtn->tuple);
    return {vtuple[gtn->index]};
  }

  NodeEntries VisitExpr_(const FunctionNode* fn) {
    ICHECK(fn->GetAttr<String>(attr::kComposite).defined())
        << "JSON runtime only supports composite functions";

    // FunctionNode should be handled by the caller.
    return {};
  }

  /*!
   * \brief Save to JSON graph
   *
   * \param writer A json writer
   */
  void Save(dmlc::JSONWriter* writer) {
    std::vector<size_t> arg_nodes;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      auto node = nodes_[i];
      if (node->IsLeaf()) {
        arg_nodes.push_back(i);
      }
    }
    size_t num_entry = 0;
    std::vector<size_t> node_row_ptr{0};
    for (auto node : nodes_) {
      num_entry += node->GetNumOutput();
      node_row_ptr.push_back(num_entry);
    }
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes_);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
    writer->WriteObjectKeyValue("heads", heads_);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->EndObject();
  }

 private:
  /*! \brief JSON graph nodes. */
  std::vector<JSONGraphObjectPtr> nodes_;
  /*! \brief Output of the JSON graph. */
  NodeEntries heads_;
  /*! \brief The list of required constants, ordered. */
  Array<String> constants_used_;
  /*! \brief The names of all constants in the original module. */
  const Map<Constant, String>& constant_names_;
};

}  // namespace contrib
}  // namespace backend
}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_
