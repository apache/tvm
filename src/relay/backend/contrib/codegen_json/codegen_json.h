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
 * \file relay/backend/contrib/codegen_json.h
 * \brief Utilities for json codegen and runtime
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_
#define TVM_RELAY_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_

#include <dmlc/any.h>
#include <dmlc/json.h>
#include <tvm/node/reflection.h>
#include <tvm/tir/op.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../../../../runtime/contrib/json/json_runtime.h"
#include "../../utils.h"

namespace tvm {
namespace relay {
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

/*! \brief Serialize a Relay expression to JSON. */
class JSONSerializer : public MemoizedExprTranslator<std::vector<JSONGraphNodeEntry>> {
 public:
  /*!
   * \brief Constructor
   *
   * \param symbol The symbol that represents the graph being converted.
   * \param expr The Relay expression to be converted to the JSON form.
   */
  JSONSerializer(std::string symbol, Expr expr)
      : symbol_(std::move(symbol)), func_(std::move(expr)) {}

  void serialize() {
    relay::Function func = Downcast<relay::Function>(func_);
    // First we convert all the parameters into input nodes.
    for (const auto& param : func->params) {
      auto node_ptr = std::make_shared<JSONGraphNode>(param->name_hint(), "input" /* op_type_ */);
      memo_[param] = AddNode(node_ptr, param);
    }
    heads_ = VisitExpr(func->body);
  }

  /*!
   * \brief Returns the accumulated map from constant names to the NDArray they must be bound to
   * at runtime. Also referred to a 'params' elsewhere in the code.
   */
  const std::unordered_map<std::string, runtime::NDArray>& const_name_to_constant() const {
    return const_name_to_constant_;
  }

  /*!
   * \brief Return the constant names in order they were encountered during translation.
   */
  const Array<String>& const_names() const { return const_names_; }

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
   * \param expr The relay expression.
   * \return A list of graph entry nodes. It the relay expr is a tuple type, we
   *         will flatten it.
   */
  std::vector<JSONGraphNodeEntry> AddNode(JSONGraphObjectPtr node, const Expr& expr) {
    auto checked_type = expr->checked_type();
    auto node_id = nodes_.size();
    nodes_.push_back(node);
    std::vector<JSONGraphNodeEntry> ret;
    ShapeVector shape;
    TypeVector dtype;
    // Flatten tuple node.
    if (const auto* tuple_type = checked_type.as<TupleTypeNode>()) {
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        const auto* tensor_type = tuple_type->fields[i].as<TensorTypeNode>();
        ICHECK(tensor_type) << "Expect TensorType, but received: ."
                            << tuple_type->fields[i]->GetTypeKey();
        ret.push_back(JSONGraphNodeEntry(node_id, i));
        shape.emplace_back(GetIntShape(tensor_type->shape));
        dtype.emplace_back(DType2String(tensor_type->dtype));
      }
      node->SetNumOutput(tuple_type->fields.size());
    } else {
      const auto* tensor_type = checked_type.as<TensorTypeNode>();
      ICHECK(tensor_type) << "Expect TensorType, but received: " << checked_type->GetTypeKey();
      shape.emplace_back(GetIntShape(tensor_type->shape));
      dtype.emplace_back(DType2String(tensor_type->dtype));
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
      auto pattern = fn->GetAttr<String>(attr::kPartitionedFromPattern);
      ICHECK(pattern.defined());
      std::vector<std::string> values;
      values.push_back(pattern.value());
      std::vector<dmlc::any> attr;
      attr.emplace_back(values);
      node->SetAttr("PartitionedFromPattern", attr);
    }
  }

  std::vector<JSONGraphNodeEntry> VisitExprDefault_(const Object* op) {
    LOG(FATAL) << "JSON runtime currently doesn't support " << op->GetTypeKey();
  }

  std::vector<JSONGraphNodeEntry> VisitExpr_(const VarNode* vn) {
    ICHECK(memo_.count(GetRef<Expr>(vn)));
    return memo_[GetRef<Expr>(vn)];
  }

  std::vector<JSONGraphNodeEntry> VisitExpr_(const ConstantNode* constant_node) {
    std::string name = symbol_ + "_const_" + std::to_string(const_names_.size());
    VLOG(1) << "Will require parameter '" << name
            << "' to be supplied by the ConstLoaderModule at runtime";
    ICHECK_EQ(const_name_to_constant_.count(name), 0);
    const_name_to_constant_.emplace(name, constant_node->data);
    const_names_.push_back(name);
    auto node = std::make_shared<JSONGraphNode>(name, /*op_type=*/"const");
    return AddNode(node, GetRef<Expr>(constant_node));
  }

  std::vector<JSONGraphNodeEntry> VisitExpr_(const TupleNode* tn) {
    std::vector<JSONGraphNodeEntry> fields;
    for (const auto& field : tn->fields) {
      auto ref = VisitExpr(field);
      fields.insert(fields.end(), ref.begin(), ref.end());
    }
    return fields;
  }

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) {
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

    std::vector<JSONGraphNodeEntry> inputs;
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

  std::vector<JSONGraphNodeEntry> VisitExpr_(const LetNode* ln) {
    ICHECK_EQ(memo_.count(ln->var), 0);
    memo_[ln->var] = VisitExpr(ln->value);
    return VisitExpr(ln->body);
  }

  std::vector<JSONGraphNodeEntry> VisitExpr_(const TupleGetItemNode* gtn) {
    auto vtuple = VisitExpr(gtn->tuple);
    return {vtuple[gtn->index]};
  }

  std::vector<JSONGraphNodeEntry> VisitExpr_(const FunctionNode* fn) {
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
    writer->WriteObjectKeyValue("symbol", symbol_);
    writer->WriteObjectKeyValue("nodes", nodes_);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
    writer->WriteObjectKeyValue("heads", heads_);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->EndObject();
  }

 private:
  /*! \brief The symbol that represents the json graph. */
  std::string symbol_;
  /*! \brief The function to be serialized. */
  const Expr func_;
  /*! \brief JSON graph nodes. */
  std::vector<JSONGraphObjectPtr> nodes_;
  /*! \brief Output of the JSON graph. */
  std::vector<JSONGraphNodeEntry> heads_;
  /*!
   * \brief A map from constant names to NDArrays for each Constant encountered during
   * translation to JSON. The JSON will record only the constant name. The actual NDArray must
   * be made available at runtime from a ConstLoaderModule.
   */
  std::unordered_map<std::string, runtime::NDArray> const_name_to_constant_;
  /*!
   * \brief The domain of the above map, but in order the constants were encountered during
   * translation.
   */
  Array<String> const_names_;
};

}  // namespace contrib
}  // namespace backend
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CODEGEN_JSON_CODEGEN_JSON_H_
