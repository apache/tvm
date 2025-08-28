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
 * \file src/relay/backend/contrib/mrvl/codegen.cc
 * \brief Marvell MLIP specific API
 */

#include <stdio.h>
#include <tvm/ir/module.h>
#include <tvm/relay/type.h>
#include <tvm/tir/analysis.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../../../../support/base64.h"
#include "../../../qnn/utils.h"
#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {

namespace contrib {
namespace mrvl {

using namespace backend;

struct const_struct {
  std::string name;
  std::string shape;
  std::string dtype;
  std::string min;
  std::string max;
  std::string data_base64;
};

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

/*!
 * \brief Generates an MrvlModule from a relay expression. This "compilation"
 * does not require Mrvl driver since the actual conversion using Mrvl APIs is
 * deferred until creation of the runtime. This step simply serializes the
 * relay program into a JSON string.
 */
class MrvlJSONSerializer : public backend::contrib::JSONSerializer {
 public:
  /*!
   * \brief Constructor
   *
   * \param symbol The symbol that represents the graph being converted.
   * \param expr The Relay expression to be converted to the JSON form.
   */
  MrvlJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {
    layer_name_ = symbol;
  }

  /*! \brief Return the required params. */
  Array<String> GetParams() const {
    tvm::runtime::Array<tvm::runtime::String> base_params = JSONSerializer::const_names();
    Array<String> mrvl_params;
    for (size_t idx = 0; idx < base_params.size(); idx++) {
      mrvl_params.push_back(base_params[idx]);
    }
    for (size_t idx = 0; idx < batch_norm_params_.size(); idx++) {
      mrvl_params.push_back(batch_norm_params_[idx]);
    }
    return mrvl_params;
  }

  template <typename T>
  std::string FloatToString(T val, size_t precision = 17) {
    // Method to serialize floating point values (double, float)
    // to a string with required precision.
    std::ostringstream s;
    s.precision(precision);
    s << val;
    return s.str();
  }

  /*! \brief Return the Const Json Strings. */
  std::string GetConstJSONString() {
    std::string json_string;
    auto names = const_names();
    auto map = const_name_to_constant();
    std::vector<const_struct> const_info_vec;
    for (auto name_const : names) {
      const_struct a;
      std::string const_string = name_const;
      auto arr = map[const_string];
      a.name = const_string;
      a.dtype = "float" + std::to_string(static_cast<int>(arr->dtype.bits));
      std::string shape;
      shape += "[ ";

      int ndim = arr->ndim;
      if (ndim == 1 || ndim == 3) {
        shape += "1, ";
      }
      int tot_dim = 1;
      for (int i = 0; i < ndim; i++) {
        tot_dim *= arr->shape[i];
        shape += std::to_string(arr->shape[i]);
        if (i != ndim - 1) shape += ", ";
      }
      shape += " ]";
      a.shape = shape;
      int size = (arr->dtype.bits + 7) / 8;
      int num_bytes = tot_dim * size;
      std::string blob;
      dmlc::MemoryStringStream mstrm(&blob);
      support::Base64OutStream b64strm(&mstrm);
      b64strm.Write(arr->data, num_bytes);
      b64strm.Finish();
      a.data_base64 = blob;
      // Populate min and max
      float min_val = std::numeric_limits<float>::infinity();
      float max_val = -min_val;
      for (int i = 0; i < tot_dim; i++) {
        auto val = static_cast<float*>(arr->data)[i];
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
      }

      a.min = FloatToString<float>(min_val);
      a.max = FloatToString<float>(max_val);

      const_info_vec.push_back(a);
    }

    json_string += "{\n";
    for (unsigned int i = 0; i < const_info_vec.size(); i++) {
      auto a = const_info_vec[i];
      json_string += "\t\"" + a.name + "\": {\n";
      json_string += "\t\"shape\": " + a.shape + ",\n";
      json_string += "\t\"dtype\": \"" + a.dtype + "\"" + ",\n";
      json_string += "\t\"min\": \"" + a.min + "\"" + ",\n";
      json_string += "\t\"max\": \"" + a.max + "\"" + ",\n";
      json_string += "\t\"data_base64\": \"" + a.data_base64 + "\"\n";
      if (i == const_info_vec.size() - 1) {
        json_string += "\t}\n";
      } else {
        json_string += "\t},\n";
      }
    }
    json_string += "}\n";
    return json_string;
  }

 protected:
  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports both nn.conv2d and qnn.conv2d.
   */
  struct CompositeConvNode {
    const CallNode* pad = nullptr;
    const CallNode* conv = nullptr;
    const CallNode* add = nullptr;
    const CallNode* batch_norm = nullptr;
    const CallNode* activation = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * sum.
   */
  struct CompositeSumNode {
    const CallNode* add = nullptr;
    const CallNode* activation = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * maxpool or avgpool. Supports both nn.max_pool2d and qnn.conv2d.
   */
  struct CompositePoolNode {
    const CallNode* pad = nullptr;
    const CallNode* pool = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * concat.
   */
  struct CompositeConcatNode {
    const CallNode* concat = nullptr;
  };

  /*!
   * \brief A series of operators that form a reshape node.
   */
  struct CompositeReshapeNode {
    const CallNode* reshape = nullptr;
  };

  /*!
   * \brief A series of operators that form a batch flatten node.
   */
  struct CompositeBatchFlattenNode {
    const CallNode* batch_flatten = nullptr;
  };

  /*!
   * \brief A series of operators that form a Squeeze node.
   */
  struct CompositeSqueezeNode {
    const CallNode* squeeze = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * fc layer. Supports both nn.fc_ni2no and qnn.fc_ni2no.
   */
  struct CompositeFcNode {
    const CallNode* transform = nullptr;
    const CallNode* flatten = nullptr;
    const CallNode* fc = nullptr;
    const CallNode* add = nullptr;
    const CallNode* activation = nullptr;
  };

  /*!
   * \brief Visit call nodes and generate appropriate JSON node.
   *
   * \param cn The current call node.
   * \return A list of graph entry nodes.
   */
  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    const auto* op_node = cn->op.as<OpNode>();
    if (op_node) {
      // handle certain op node types specially
      String op_name = op_node->name;
      bool handle_by_mrvl = (op_name == "layout_transform" || op_name == "transpose");
      if (!handle_by_mrvl) {
        return JSONSerializer::VisitExpr_(cn);
      }

      // setup json attributes and then add the Mrvl Layer to JSON files
      std::shared_ptr<JSONGraphNode> json_kernel_node;
      json_kernel_node = CreateMrvlLayer4OpNode(cn);
      return AddNode(json_kernel_node, GetRef<Expr>(cn));
    }

    // handle only mrvl composite functions
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "Mrvl JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "Marvell-Compiler-ERROR-Internal::Illegal Mrvl composite function.";
    const std::string name = comp.value();
    std::shared_ptr<JSONGraphNode> json_kernel_node;
    if (name == "mrvl.conv2d_nhwc2nhwc") {
      json_kernel_node = CreateCompositeMrvlConv2DLayer(cn);
    } else if (name == "mrvl.fc_ni2no") {
      json_kernel_node = CreateCompositeMrvlFcLayer(cn);
    } else if (name == "mrvl.maxpool2d_nhwc2nhwc") {
      json_kernel_node = CreateCompositeMrvlMaxpool2DLayer(cn);
    } else if (name == "mrvl.avgpool2d_nhwc2nhwc") {
      json_kernel_node = CreateCompositeMrvlAvgpool2DLayer(cn);
    } else if (name == "mrvl.globalavgpool2d_nhwc2nhwc") {
      json_kernel_node = CreateCompositeMrvlGlobalAvgpool2DLayer(cn);
    } else if (name == "mrvl.globalmaxpool2d_nhwc2nhwc") {
      json_kernel_node = CreateCompositeMrvlGlobalMaxpool2DLayer(cn);
    } else if (name == "mrvl.sum") {
      json_kernel_node = CreateCompositeMrvlSumLayer(cn);
    } else if (name == "mrvl.concat") {
      json_kernel_node = CreateMrvlConcatLayer(cn);
    } else if (name == "mrvl.reshape") {
      json_kernel_node = CreateMrvlReshapeLayer(cn);
    } else if (name == "mrvl.batch_flatten") {
      json_kernel_node = CreateMrvlBatchFlattenLayer(cn);
    } else if (name == "mrvl.squeeze") {
      json_kernel_node = CreateMrvlSqueezeLayer(cn);
    } else {
      LOG(FATAL) << "Unrecognized Mrvl pattern: " << name;
    }
    // calling codegen_json.h::AddNode()
    return AddNode(json_kernel_node, GetRef<Expr>(cn));
  }

 private:
  /*! \brief The symbol that represents the layer json graph. */
  std::string layer_name_;
  Array<String> batch_norm_params_;
  int node_idx_{0};
  int const_suffix_{0};

  void resizeInputOutputLayoutTo4dim(std::shared_ptr<JSONGraphNode> json_node, const CallNode* cn,
                                     std::string node_name) {
    const uint64_t new_layout_size = 4;
    std::string data_layout = "NHWC";
    std::string out_layout = "NHWC";

    auto num_inputs = GetInputNum(cn);
    auto num_outputs = GetOutputNum(cn);
    uint64_t max_old_input_layout_size = 0;
    // Inputs
    if (num_inputs > 1) {
      for (uint64_t in_idx = 0; in_idx < num_inputs; in_idx++) {
        std::vector<int64_t> layout;
        GetInputTensorShapeViaArgN(cn, &layout, in_idx);
        uint64_t old_layout_size = layout.size();
        max_old_input_layout_size = std::max(old_layout_size, max_old_input_layout_size);
        ICHECK(old_layout_size <= 4) << "Marvell-Compiler-ERROR-Internal::" << node_name
                                     << " with input tensor shape > 4 is not supported yet.";
        layout.resize(new_layout_size, 1);

        if (!cn->args[in_idx].as<ConstantNode>()) {
          JsonNodeSetVecAttr(json_node, "data_layout_shape_" + std::to_string(in_idx), layout);
          if (in_idx == 0) {
            JsonNodeSetVecAttr(json_node, "data_layout_shape", layout);
          }
        }
      }
      for (uint64_t in_idx = 0; in_idx < num_inputs; in_idx++) {
        std::vector<int64_t> layout;
        GetInputTensorShapeViaArgN(cn, &layout, in_idx);
        uint64_t old_layout_size = layout.size();
        ICHECK(old_layout_size <= 4) << "Marvell-Compiler-ERROR-Internal::" << node_name
                                     << " with input tensor shape > 4 is not supported yet.";
        layout.resize(max_old_input_layout_size, 1);
        std::rotate(layout.begin(), layout.end() - (max_old_input_layout_size - old_layout_size),
                    layout.end());
        layout.resize(new_layout_size, 1);
        if (cn->args[in_idx].as<ConstantNode>()) {
          std::vector<std::string> const_name = {layer_name_ + "_const_" +
                                                 std::to_string(const_suffix_++)};
          JsonNodeSetAttr(json_node, "input_const_name", const_name);
          JsonNodeSetVecAttr(json_node, "input_const_shape", layout);
        }
      }
    } else {
      std::vector<int64_t> layout;
      GetInputTensorShapeViaArgN(cn, &layout, 0);
      layout.resize(new_layout_size, 1);
      JsonNodeSetVecAttr(json_node, "data_layout_shape", layout);
    }
    // Outputs
    if (num_outputs > 1) {
      std::vector<std::vector<int64_t>> layout;
      GetOutputTensorShapes(cn, &layout);
      for (size_t out_idx = 0; out_idx < num_outputs; out_idx++) {
        ICHECK(layout.at(out_idx).size() <= 4)
            << "Marvell-Compiler-ERROR-Internal::" << node_name
            << " with output tensor shape > 4 is not supported yet.";
        layout.at(out_idx).resize(new_layout_size, 1);
        JsonNodeSetVecAttr(json_node, "out_layout_shape_" + std::to_string(out_idx),
                           layout.at(out_idx));
        if (out_idx == 0) {
          JsonNodeSetVecAttr(json_node, "out_layout_shape", layout.at(out_idx));
        }
      }
    } else {
      std::vector<int64_t> layout;
      GetOutputTensorShape(cn, &layout);
      layout.resize(new_layout_size, 1);
      JsonNodeSetVecAttr(json_node, "out_layout_shape", layout);
    }

    std::vector<std::string> layout_format_vec = {data_layout};
    JsonNodeSetAttr(json_node, "data_layout", layout_format_vec);
    JsonNodeSetAttr(json_node, "out_layout", layout_format_vec);
  }

  /*!
   * \brief Extract convolution nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  CompositeConvNode UnpackCompositeConvolution(const CallNode* call) {
    CompositeConvNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";
    // - conv2d + [ bias_add ] + [ batch_norm + tuple.getitem(0) ] + [ relu ]
    // Traverse composite convolution function from child to parent
    const TupleGetItemNode* tuple_get_item_node = nullptr;
    const CallNode* current_call = fn->body.as<CallNode>();
    if (current_call) {
      if (backend::IsOp(current_call, "nn.relu")) {
        nodes.activation = current_call;
        if (current_call->args[0].as<TupleGetItemNode>()) {
          tuple_get_item_node = current_call->args[0].as<TupleGetItemNode>();
        } else {
          current_call = current_call->args[0].as<CallNode>();
        }
      } else {
        ICHECK(current_call) << "Marvell-Compiler-ERROR-Internal::Downcast to CallNode failed.";
      }
    } else {
      tuple_get_item_node = fn->body.as<TupleGetItemNode>();
    }

    if (tuple_get_item_node != nullptr) {
      ICHECK(tuple_get_item_node->index == 0)
          << "Marvell-Compiler-ERROR-Internal::(index == 0) failed for the TupleGetItem node.";
      current_call = tuple_get_item_node->tuple.as<CallNode>();

      ICHECK(backend::IsOp(current_call, "nn.batch_norm"))
          << "Marvell-Compiler-ERROR-Internal::nn.batch_norm Op missing.";
      nodes.batch_norm = current_call;
      current_call = nodes.batch_norm->args[0].as<CallNode>();
    }

    ICHECK(current_call) << "Marvell-Compiler-ERROR-Internal::Downcast to CallNode failed.";
    if (backend::IsOp(current_call, "add")) {
      nodes.add = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    ICHECK(backend::IsOp(current_call, "nn.conv2d"))
        << "Marvell-Compiler-ERROR-Internal::nn.conv2d Op missing.";
    nodes.conv = current_call;
    current_call = current_call->args[0].as<CallNode>();

    if (current_call && backend::IsOp(current_call, "nn.pad")) {
      nodes.pad = current_call;
    }
    return nodes;
  }

  /*!
   * \brief Extract sum nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite sum nodes.
   */
  CompositeSumNode UnpackCompositeSum(const CallNode* call) {
    CompositeSumNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";

    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    ICHECK(backend::IsOp(current_call, "add"))
        << "Marvell-Compiler-ERROR-Internal::add Op missing.";
    nodes.add = current_call;

    return nodes;
  }

  /*!
   * \brief Extract Concat nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite Concat nodes.
   */
  CompositeConcatNode UnpackCompositeConcat(const CallNode* call) {
    CompositeConcatNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";

    const auto* current_call = fn->body.as<CallNode>();

    ICHECK(backend::IsOp(current_call, "concatenate"))
        << "Marvell-Compiler-ERROR-Internal::concatenate Op missing.";
    nodes.concat = current_call;

    return nodes;
  }

  /*!
   * \brief Extract Reshape nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite Reshape nodes.
   */
  CompositeReshapeNode UnpackCompositeReshape(const CallNode* call) {
    CompositeReshapeNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";
    const auto* current_call = fn->body.as<CallNode>();
    ICHECK(backend::IsOp(current_call, "reshape"))
        << "Marvell-Compiler-ERROR-Internal::reshape missing.";
    nodes.reshape = current_call;
    return nodes;
  }

  /*!
   * \brief Extract Batch flatten nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite batch flatten nodes.
   */
  CompositeBatchFlattenNode UnpackCompositeBatchFlatten(const CallNode* call) {
    CompositeBatchFlattenNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";
    const auto* current_call = fn->body.as<CallNode>();
    ICHECK(backend::IsOp(current_call, "nn.batch_flatten"))
        << "Marvell-Compiler-ERROR-Internal::batch_flatten missing.";
    nodes.batch_flatten = current_call;
    return nodes;
  }

  /*!
   * \brief Extract squeeze nodes from a composite function.
   * \param call The call node of the composite function.
   * \return Extracted composite squeeze nodes.
   */
  CompositeSqueezeNode UnpackCompositeSqueeze(const CallNode* call) {
    CompositeSqueezeNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";
    const auto* current_call = fn->body.as<CallNode>();
    ICHECK(backend::IsOp(current_call, "squeeze"))
        << "Marvell-Compiler-ERROR-Internal::squeeze missing.";
    nodes.squeeze = current_call;
    return nodes;
  }

  /*!
   * \brief Extract maxpool nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite maxpool nodes.
   */
  CompositePoolNode UnpackCompositePool(const CallNode* call, const std::string& mrvlLayerName) {
    CompositePoolNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";

    // Traverse composite maxpool function from child to parent
    const auto* current_call = fn->body.as<CallNode>();

    if (mrvlLayerName == "Maxpool2D") {
      ICHECK(backend::IsOp(current_call, "nn.max_pool2d"))
          << "Marvell-Compiler-ERROR-Internal::nn.max_pool2d Op missing.";
    } else if (mrvlLayerName == "Avgpool2D") {
      ICHECK(mrvlLayerName == "Avgpool2D")
          << "Marvell-Compiler-ERROR-Internal::nn.avg_pool2d Op missing.";
      ICHECK(backend::IsOp(current_call, "nn.avg_pool2d"))
          << "Marvell-Compiler-ERROR-Internal::nn.avg_pool2d Op missing.";
    } else if (mrvlLayerName == "GlobalMaxpool2D") {
      ICHECK(mrvlLayerName == "GlobalMaxpool2D")
          << "Marvell-Compiler-ERROR-Internal::nn.global_max_pool2d Op missing.";
      ICHECK(backend::IsOp(current_call, "nn.global_max_pool2d"))
          << "Marvell-Compiler-ERROR-Internal::nn.global_max_pool2d Op missing.";
    } else {
      ICHECK(mrvlLayerName == "GlobalAvgpool2D")
          << "Marvell-Compiler-ERROR-Internal::nn.global_avg_pool2d Op missing.";
      ICHECK(backend::IsOp(current_call, "nn.global_avg_pool2d"))
          << "Marvell-Compiler-ERROR-Internal::nn.global_avg_pool2d Op missing.";
    }
    nodes.pool = current_call;
    current_call = current_call->args[0].as<CallNode>();
    if (current_call && backend::IsOp(current_call, "nn.pad")) {
      nodes.pad = current_call;
    }

    return nodes;
  }

  /*!
   * \brief Extract fc nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite fc nodes.
   */
  CompositeFcNode UnpackCompositeFc(const CallNode* call) {
    CompositeFcNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn) << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";
    const auto* current_call = fn->body.as<CallNode>();

    // Traverse composite fc function from child to parent
    if (backend::IsOp(current_call, "nn.batch_flatten")) {
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "add")) {
      nodes.add = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    ICHECK(backend::IsOp(current_call, "nn.dense"))
        << "Marvell-Compiler-ERROR-Internal::nn.dense Op missing.";
    nodes.fc = current_call;
    current_call = current_call->args[0].as<CallNode>();
    if (current_call) {
      if (backend::IsOp(current_call, "reshape") |
          backend::IsOp(current_call, "nn.batch_flatten")) {
        nodes.flatten = current_call;
        current_call = current_call->args[0].as<CallNode>();
        ICHECK(backend::IsOp(current_call, "layout_transform"))
            << "Marvell-Compiler-ERROR-Internal::layout_transform Op missing.";
        nodes.transform = current_call;
      }
    }

    return nodes;
  }

  void JsonNodeSetAttr(std::shared_ptr<JSONGraphNode> json_node, const std::string& key,
                       const std::vector<std::string>& string_vec) {
    std::vector<dmlc::any> json_attr;
    json_attr.emplace_back(string_vec);
    json_node->SetAttr(key, json_attr);
  }

  void JsonNodeSetVecAttr(std::shared_ptr<JSONGraphNode> json_node, const std::string& key,
                          const std::vector<int64_t>& tvec) {
    size_t tvec_size = tvec.size();
    std::vector<std::string> tvec_str;
    if (tvec_size == 4) {
      tvec_str = {std::to_string(tvec[0]), std::to_string(tvec[1]), std::to_string(tvec[2]),
                  std::to_string(tvec[3])};
    } else if (tvec_size == 3) {
      tvec_str = {std::to_string(tvec[0]), std::to_string(tvec[1]), std::to_string(tvec[2])};
    } else if (tvec_size == 2) {
      tvec_str = {std::to_string(tvec[0]), std::to_string(tvec[1])};
    } else {
      tvec_str = {std::to_string(tvec[0])};
    }
    std::vector<dmlc::any> json_attr;
    json_attr.emplace_back(tvec_str);
    json_node->SetAttr(key, json_attr);
  }

  void SetMrvlLayerBatchnormAttrs(std::shared_ptr<JSONGraphNode> json_node,
                                  const CallNode* cn_batchnorm) {
    if (cn_batchnorm == nullptr) return;

    SetCallNodeAttribute(json_node, cn_batchnorm);

    std::vector<std::string> gamma_const_name;
    std::vector<std::string> beta_const_name;
    std::vector<std::string> mean_const_name;
    std::vector<std::string> var_const_name;
    std::string batch_norm_layout = "-O";

    gamma_const_name = {layer_name_ + "_const_" + std::to_string(const_suffix_++)};
    beta_const_name = {layer_name_ + "_const_" + std::to_string(const_suffix_++)};
    mean_const_name = {layer_name_ + "_const_" + std::to_string(const_suffix_++)};
    var_const_name = {layer_name_ + "_const_" + std::to_string(const_suffix_++)};

    JsonNodeSetAttr(json_node, "gamma_const_name", gamma_const_name);
    JsonNodeSetAttr(json_node, "beta_const_name", beta_const_name);
    JsonNodeSetAttr(json_node, "mean_const_name", mean_const_name);
    JsonNodeSetAttr(json_node, "var_const_name", var_const_name);
    JsonNodeSetAttr(json_node, "gamma_layout", {batch_norm_layout});
    JsonNodeSetAttr(json_node, "beta_layout", {batch_norm_layout});
    JsonNodeSetAttr(json_node, "mean_layout", {batch_norm_layout});
    JsonNodeSetAttr(json_node, "var_layout", {batch_norm_layout});
  }

  void SetMrvlLayerPadAttrs(std::shared_ptr<JSONGraphNode> json_node, const CallNode* cn_pad) {
    if (cn_pad == nullptr) return;

    const auto* pad_attr = cn_pad->attrs.as<PadAttrs>();
    ICHECK(pad_attr) << "Marvell-Compiler-ERROR-Internal::Downcast to PadAttrs failed.";
    ICHECK(cn_pad->args[1].as<ConstantNode>() == 0)
        << "Marvell-Compiler-ERROR-Internal::padded value is non-zero.";
    ICHECK(pad_attr->pad_mode == "constant")
        << "Marvell-Compiler-ERROR-Internal::unsupported padding mode.";

    auto p = pad_attr->pad_width;
    // Convert to TVM layout for now, conversion to Mrvl layout takes place in runtime.
    // Standard pad layout for TVM: top, left, bottom, right.
    std::vector<std::string> padding = {std::to_string(p[1][0].as<IntImmNode>()->value),
                                        std::to_string(p[2][0].as<IntImmNode>()->value),
                                        std::to_string(p[1][1].as<IntImmNode>()->value),
                                        std::to_string(p[2][1].as<IntImmNode>()->value)};

    JsonNodeSetAttr(json_node, "padding", {padding});
  }

  void SetMrvlLayerCommonAttrs(std::shared_ptr<JSONGraphNode> json_node, const CallNode* cn,
                               const std::string& func_name, const std::string& mrvlLayerName,
                               const std::string& data_layout, const std::string& kernel_layout,
                               const std::string& out_layout) {
    JsonNodeSetAttr(json_node, "layer_name", {mrvlLayerName});
    JsonNodeSetAttr(json_node, "func_node_name", {func_name});
    std::vector<int64_t> data_layout_vec;

    auto num_inputs = GetInputNum(cn);
    auto num_outputs = GetOutputNum(cn);
    auto counter = num_inputs;
    for (size_t i = 0; i < counter; i++) {
      if (cn->args[i].as<ConstantNode>()) num_inputs--;
    }

    std::vector<int64_t> tuple_idx_vec;
    int tuple_idx = -1;
    if (num_inputs > 1) {
      for (size_t in_idx = 0; in_idx < num_inputs; in_idx++) {
        std::vector<int64_t> data_layout_vec_n;
        tuple_idx = GetInputTensorShapeViaArgN(cn, &data_layout_vec_n, in_idx);
        std::string attr_name = "data_layout_shape_" + std::to_string(in_idx);
        JsonNodeSetVecAttr(json_node, attr_name, data_layout_vec_n);
        tuple_idx_vec.push_back(tuple_idx);
        if (in_idx == 0) {
          JsonNodeSetVecAttr(json_node, "data_layout_shape", data_layout_vec_n);
        }
      }
    } else {
      tuple_idx = GetInputTensorShapeViaArgN(cn, &data_layout_vec, 0);
      JsonNodeSetVecAttr(json_node, "data_layout_shape", data_layout_vec);
      tuple_idx_vec.push_back(tuple_idx);
    }
    JsonNodeSetVecAttr(json_node, "from_tuple_idx", tuple_idx_vec);

    if (data_layout != "") {
      std::vector<std::string> data_layout_format_vec = {data_layout};
      JsonNodeSetAttr(json_node, "data_layout", data_layout_format_vec);
    }

    std::vector<int64_t> out_layout_vec;
    if (num_outputs > 1) {
      std::vector<std::vector<int64_t>> output_layout_vec_vec;
      GetOutputTensorShapes(cn, &output_layout_vec_vec);
      for (size_t out_idx = 0; out_idx < num_outputs; out_idx++) {
        std::string attr_name = "out_layout_shape_" + std::to_string(out_idx);
        JsonNodeSetVecAttr(json_node, attr_name, output_layout_vec_vec.at(out_idx));
      }
      // For compatibility with backend
      JsonNodeSetVecAttr(json_node, "out_layout_shape", output_layout_vec_vec.at(0));
    } else {
      GetOutputTensorShape(cn, &out_layout_vec);
      JsonNodeSetVecAttr(json_node, "out_layout_shape", out_layout_vec);
    }

    if (kernel_layout != "") {
      std::vector<std::string> kernel_layout_format_vec = {kernel_layout};
      JsonNodeSetAttr(json_node, "kernel_layout", kernel_layout_format_vec);
    }
    if (out_layout != "") {
      std::vector<std::string> out_layout_format_vec = {out_layout};
      JsonNodeSetAttr(json_node, "out_layout", out_layout_format_vec);
    }

    // setup n<#>_<mrvlLayerName> as GUI node name ("func_name") in nodes JSON file
    std::string node_id_func_name = "";
    node_id_func_name = "n" + std::to_string(node_idx_++) + "_" + mrvlLayerName;

    // - add posfix layout(s) if applicable
    if ((data_layout != "") && (out_layout != "")) {
      node_id_func_name += "_" + data_layout;
      if (data_layout != out_layout) {
        node_id_func_name += "2" + out_layout;
      }
    }

    JsonNodeSetAttr(json_node, "func_name", {node_id_func_name});

    const auto* fn = cn->op.as<FunctionNode>();
    if (fn != nullptr) {
      ICHECK(fn->IsInstance<FunctionNode>())
          << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";
      auto composite = fn->GetAttr<String>(attr::kComposite);
      ICHECK(composite.defined())
          << "Marvell-Compiler-ERROR-Internal::Illegal Mrvl composite function.";
      std::string composite_name = composite.value();
      JsonNodeSetAttr(json_node, "composite_name", {composite_name});
    }
  }

  void GetInputTensorShapeFromTuple(const CallNode* call_node_ptr, size_t index,
                                    std::vector<int64_t>* tensor_shape) {
    ICHECK(!call_node_ptr->args.empty());
    const TensorTypeNode* tensor_type = nullptr;
    if (call_node_ptr->args[0].as<CallNode>()) {
      const auto* arg0 = call_node_ptr->args[0].as<CallNode>();
      tensor_type = arg0->checked_type_.as<TensorTypeNode>();
    } else if (call_node_ptr->args[0].as<VarNode>()) {
      const auto* arg0 = call_node_ptr->args[0].as<VarNode>();
      ICHECK((arg0 != nullptr) && arg0->IsInstance<VarNode>())
          << "Marvell-Compiler-ERROR-Internal::Downcast to VarNode failed.";
      tensor_type = arg0->checked_type_.as<TensorTypeNode>();
      const TupleTypeNode* tuple_type = arg0->checked_type_.as<TupleTypeNode>();
      if (tuple_type) {
        tensor_type = tuple_type->fields[index].as<TensorTypeNode>();
      }
    } else {
      LOG(INFO) << "TVM Mrvl runtime does not support calls to "
                << call_node_ptr->args[0]->GetTypeKey();
    }

    ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>())
        << "Marvell-Compiler-ERROR-Internal::Downcast to TensorTypeNode failed.";
    for (IndexExpr dim_val : tensor_type->shape) {
      tensor_shape->push_back(*(tir::as_const_int(dim_val)));
    }
  }

  size_t GetInputNum(const CallNode* call_node_ptr) {
    size_t num_inputs = call_node_ptr->args.size();
    ICHECK(!call_node_ptr->args.empty());
    const TupleGetItemNode* tuple_get_item_node = call_node_ptr->args[0].as<TupleGetItemNode>();
    const TensorTypeNode* tensor_type = nullptr;
    if (tuple_get_item_node) {
      tensor_type = tuple_get_item_node->checked_type().as<TensorTypeNode>();
    } else if (call_node_ptr->args[0].as<CallNode>()) {
      num_inputs = call_node_ptr->args.size();
    } else if (call_node_ptr->args[0].as<VarNode>()) {
      const auto* arg_0 = call_node_ptr->args[0].as<VarNode>();
      ICHECK((arg_0 != nullptr) && arg_0->IsInstance<VarNode>())
          << "Marvell-Compiler-ERROR-Internal::Downcast to VarNode failed.";
      tensor_type = arg_0->checked_type_.as<TensorTypeNode>();
      if (tensor_type == nullptr) {
        const TupleTypeNode* tuple_type = arg_0->checked_type_.as<TupleTypeNode>();
        if (tuple_type) {
          num_inputs = tuple_type->fields.size();
        }
      }
    } else {
      LOG(INFO) << "TVM Mrvl runtime does not support calls to "
                << call_node_ptr->args[0]->GetTypeKey();
    }
    return num_inputs;
  }

  size_t GetOutputNum(const CallNode* call_node_ptr) {
    ICHECK(call_node_ptr != nullptr);
    const TupleTypeNode* tuple_type = call_node_ptr->checked_type_.as<TupleTypeNode>();
    if (tuple_type) {
      return tuple_type->fields.size();
    }
    // If output isn't a tuple, there is a single output
    return 1;
  }

  void GetInputTensorShapeViaArg(const CallNode* call_node_ptr, std::vector<int64_t>* tensor_shape,
                                 int* tuple_index, size_t n) {
    *tuple_index = -1;
    ICHECK(!call_node_ptr->args.empty());
    const TensorTypeNode* tensor_type = nullptr;
    const TupleGetItemNode* tuple_get_item_node = call_node_ptr->args[n].as<TupleGetItemNode>();
    if (tuple_get_item_node) {
      *tuple_index = tuple_get_item_node->index;
      tensor_type = tuple_get_item_node->checked_type().as<TensorTypeNode>();
    } else if (call_node_ptr->args[n].as<CallNode>()) {
      const auto* arg_n = call_node_ptr->args[n].as<CallNode>();
      tensor_type = arg_n->checked_type().as<TensorTypeNode>();
    } else if (call_node_ptr->args[n].as<VarNode>()) {
      const auto* arg_n = call_node_ptr->args[n].as<VarNode>();
      ICHECK((arg_n != nullptr) && arg_n->IsInstance<VarNode>())
          << "Marvell-Compiler-ERROR-Internal::Downcast to VarNode failed.";
      tensor_type = arg_n->checked_type().as<TensorTypeNode>();
      if (tensor_type == nullptr) {
        const TupleTypeNode* tuple_type = arg_n->checked_type().as<TupleTypeNode>();
        if (tuple_type) {
          tensor_type = tuple_type->fields[n].as<TensorTypeNode>();
        }
      } else if (call_node_ptr->args[n].as<ConstantNode>()) {
        const auto* arg_n = call_node_ptr->args[n].as<ConstantNode>();
        ICHECK((arg_n != nullptr) && arg_n->IsInstance<ConstantNode>())
            << "Marvell-Compiler-ERROR-Internal::Downcast to ConstantNode failed.";
        tensor_type = arg_n->checked_type().as<TensorTypeNode>();
        if (tensor_type == nullptr) {
          const TupleTypeNode* tuple_type = arg_n->checked_type().as<TupleTypeNode>();
          if (tuple_type) {
            tensor_type = tuple_type->fields[n].as<TensorTypeNode>();
          }
        }
      }
    } else {
      LOG(INFO) << "TVM Mrvl runtime does not support calls to "
                << call_node_ptr->args[n]->GetTypeKey();
    }

    ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>())
        << "Marvell-Compiler-ERROR-Internal::Downcast to TensorTypeNode failed.";
    // use only data types supported by json.h (e.g., int or int64_t or size_t)
    for (IndexExpr dim_val : tensor_type->shape) {
      tensor_shape->push_back(*(tir::as_const_int(dim_val)));
    }
  }

  int GetInputTensorShapeViaArgN(const CallNode* call_node_ptr, std::vector<int64_t>* tensor_shape,
                                 int64_t n = 0) {
    int tuple_idx = -1;
    GetInputTensorShapeViaArg(call_node_ptr, tensor_shape, &tuple_idx, n);
    return tuple_idx;
  }

  void GetTensorShape(const VarNode* var_node_ptr, std::vector<int64_t>* tensor_shape) {
    ICHECK((var_node_ptr != nullptr) && var_node_ptr->IsInstance<VarNode>())
        << "Marvell-Compiler-ERROR-Internal::Downcast to VarNode failed.";
    const TensorTypeNode* tensor_type = var_node_ptr->checked_type_.as<TensorTypeNode>();
    ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>())
        << "Marvell-Compiler-ERROR-Internal::Downcast to TensorTypeNode failed.";
    // use only data types supported by json.h (e.g., int or int64_t or size_t)
    for (IndexExpr dim_val : tensor_type->shape) {
      tensor_shape->push_back(*(tir::as_const_int(dim_val)));
    }
  }

  void GetOutputTensorShape(const CallNode* call_node_ptr, std::vector<int64_t>* tensor_shape) {
    ICHECK(call_node_ptr != nullptr);
    const TensorTypeNode* tensor_type = call_node_ptr->checked_type_.as<TensorTypeNode>();
    ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>())
        << "Marvell-Compiler-ERROR-Internal::Downcast to TensorTypeNode failed.";
    for (IndexExpr dim_val : tensor_type->shape) {
      tensor_shape->push_back(*(tir::as_const_int(dim_val)));
    }
  }

  void GetOutputTensorShapes(const CallNode* call_node_ptr,
                             std::vector<std::vector<int64_t>>* tensor_shapes) {
    ICHECK(call_node_ptr != nullptr);

    const TupleTypeNode* tuple_type = call_node_ptr->checked_type_.as<TupleTypeNode>();
    ICHECK((tuple_type != nullptr) && tuple_type->IsInstance<TupleTypeNode>())
        << "Marvell-Compiler-ERROR-Internal::Downcast to TupleTypeNode failed.";
    for (auto field : tuple_type->fields) {
      const TensorTypeNode* tensor_type = field.as<TensorTypeNode>();
      ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>())
          << "Marvell-Compiler-ERROR-Internal::Downcast to TensorTypeNode failed.";
      // use only data types supported by json.h (e.g., int or int64_t or size_t)
      std::vector<int64_t> tensor_shape;
      for (IndexExpr dim_val : tensor_type->shape) {
        tensor_shape.push_back(*(tir::as_const_int(dim_val)));
      }
      tensor_shapes->push_back(tensor_shape);
    }
  }

  /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlConv2DLayer(const CallNode* cn) {
    CompositeConvNode nodes = UnpackCompositeConvolution(cn);
    const auto* conv_attrs = nodes.conv->attrs.as<Conv2DAttrs>();
    ICHECK(conv_attrs) << "Marvell-Compiler-ERROR-Internal::Downcast to Conv2DAttrs failed.";

    std::string name;
    std::string mrvlLayerName = "";
    std::string data_layout;
    std::string kernel_layout;
    std::string out_layout;
    std::vector<JSONGraphNodeEntry> inputs;

    // data input tensor
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    // weight tensor
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    if (nodes.add) {
      // bias tensor
      inputs.push_back(VisitExpr(nodes.add->args[1])[0]);
    }
    if (nodes.batch_norm) {
      // get gamma, beta, mean, and var of batch-norm
      for (size_t const_idx = 0; const_idx <= 3; const_idx++) {
        size_t arg_idx = const_idx + 1;
        ICHECK(nodes.batch_norm->args[arg_idx].as<ConstantNode>())
            << "Marvell-Compiler-ERROR-Internal::Downcast to ConstantNode failed.";
        auto n = nodes.batch_norm->args[arg_idx];
        auto it = memo_.find(n);
        if (it != memo_.end()) {
          memo_.erase(n);
        }
        inputs.push_back(VisitExpr(n)[0]);
      }
    }

    // Distinguish between normal and depth-wise convolution
    data_layout = conv_attrs->data_layout;
    kernel_layout = conv_attrs->kernel_layout;
    out_layout = conv_attrs->out_layout;
    int groups = conv_attrs->groups;
    if ((groups != 1) && conv_attrs->channels.defined() &&
        tvm::tir::ExprDeepEqual()(conv_attrs->channels, conv_attrs->groups)) {
      name = "nn.dw_conv2d_nhwc2nhwc";
      mrvlLayerName = "Conv2D";
      if (conv_attrs->groups == 1) {
        ICHECK(kernel_layout == "IHWO")
            << "Marvell-Compiler-ERROR-Internal::"
            << "Kernel layout must be IHWO, has the module been pre-processed correctly?";
      }
    } else {
      name = "nn.conv2d_nhwc2nhwc";
      mrvlLayerName = "Conv2D";
      ICHECK(data_layout == "NHWC")
          << "Marvell-Compiler-ERROR-Internal::"
          << "Data layout must be NHWC, has the module been pre-processed correctly?";
      ICHECK(kernel_layout == "OHWI")
          << "Marvell-Compiler-ERROR-Internal::"
          << "Kernel layout must be OHWI, has the module been pre-processed correctly?";
      ICHECK(out_layout == "NHWC")
          << "Marvell-Compiler-ERROR-Internal::"
          << "Out layout must be NHWC, has the module been pre-processed correctly?";
    }

    // add json node attributes
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);
    std::vector<std::string> kernel_const_name = {layer_name_ + "_const_" +
                                                  std::to_string(const_suffix_++)};
    JsonNodeSetAttr(json_node, "kernel_const_name", kernel_const_name);

    if (nodes.add) {
      SetCallNodeAttribute(json_node, nodes.add);
      std::vector<std::string> bias_const_name = {layer_name_ + "_const_" +
                                                  std::to_string(const_suffix_++)};
      JsonNodeSetAttr(json_node, "bias_const_name", bias_const_name);
      JsonNodeSetAttr(json_node, "bias_layout", {"---O"});
    }
    if (nodes.pad) SetMrvlLayerPadAttrs(json_node, nodes.pad);
    if (nodes.batch_norm) SetMrvlLayerBatchnormAttrs(json_node, nodes.batch_norm);
    if (nodes.activation) JsonNodeSetAttr(json_node, "activation_type", {"relu"});
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, "", out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite sum.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlSumLayer(const CallNode* cn) {
    CompositeSumNode nodes = UnpackCompositeSum(cn);
    ICHECK(nodes.add != nullptr)
        << "Marvell-Compiler-ERROR-Internal::attribute add can't be nullptr";

    std::string mrvlLayerName = "Sum2D";
    std::string name = "sum";
    std::string data_layout;
    std::string out_layout;
    std::vector<int64_t> layout_vec;
    std::vector<JSONGraphNodeEntry> inputs;

    for (auto arg : cn->args) {
      inputs.push_back(VisitExpr(arg)[0]);
    }

    // add json node attributes
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.add);
    if (nodes.activation) JsonNodeSetAttr(json_node, "activation_type", {"relu"});
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, "", out_layout);
    resizeInputOutputLayoutTo4dim(json_node, cn, "Sum");
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite reshape.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateMrvlReshapeLayer(const CallNode* cn) {
    CompositeReshapeNode nodes = UnpackCompositeReshape(cn);

    std::string name = "reshape";
    std::string data_layout;
    std::string out_layout;
    std::vector<int64_t> layout_vec;
    std::vector<JSONGraphNodeEntry> inputs;

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    GetInputTensorShapeViaArgN(nodes.reshape, &layout_vec);
    ICHECK(layout_vec.size() == 2 || layout_vec.size() == 4)
        << "Marvell-Compiler-ERROR-Internal::"
        << "Reshape with input tensor dim != 2 or != 4 is not supported yet.";
    if (layout_vec.size() == 4) {
      data_layout = "NHWC";
    } else {
      data_layout = "NC";
    }
    layout_vec.clear();
    GetOutputTensorShape(cn, &layout_vec);
    ICHECK(layout_vec.size() == 2 || layout_vec.size() == 4)
        << "Marvell-Compiler-ERROR-Internal::"
        << "Reshape with output tensor dim != 2 or !=4 is not supported yet.";
    if (layout_vec.size() == 4) {
      out_layout = "NHWC";
    } else {
      out_layout = "NC";
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, name, data_layout,
                            "" /* no kernel_layout */, out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite batch flatten.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateMrvlBatchFlattenLayer(const CallNode* cn) {
    CompositeBatchFlattenNode nodes = UnpackCompositeBatchFlatten(cn);

    std::string name = "nn.batch_flatten";
    std::string data_layout;
    std::string out_layout = "NC";
    std::vector<int64_t> layout_vec;
    std::vector<JSONGraphNodeEntry> inputs;

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    GetInputTensorShapeViaArgN(nodes.batch_flatten, &layout_vec);
    ICHECK(layout_vec.size() == 2 || layout_vec.size() == 4)
        << "Marvell-Compiler-ERROR-Internal::"
        << "nn.batch_flatten with input tensor dim != 2 or != 4 is not supported yet.";
    if (layout_vec.size() == 4) {
      data_layout = "NHWC";
    } else {
      data_layout = "NC";
    }
    layout_vec.clear();
    GetOutputTensorShape(cn, &layout_vec);
    ICHECK(layout_vec.size() == 2)
        << "Marvell-Compiler-ERROR-Internal::"
        << "nn.batch_flatten with output tensor dim != 2 is not supported yet.";

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, name, data_layout,
                            "" /* no kernel_layout */, out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite Squeeze.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateMrvlSqueezeLayer(const CallNode* cn) {
    CompositeSqueezeNode nodes = UnpackCompositeSqueeze(cn);
    std::vector<JSONGraphNodeEntry> inputs;
    std::string name = "squeeze";
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    std::vector<int64_t> layout_vec;
    GetInputTensorShapeViaArgN(nodes.squeeze, &layout_vec);
    std::string data_layout;
    if (layout_vec.size() == 4) {
      data_layout = "NHWC";
    } else {
      data_layout = "NC";
    }
    layout_vec.clear();
    std::string out_layout = "NC";
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, name, data_layout,
                            "" /* no kernel_layout */, out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite concat.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateMrvlConcatLayer(const CallNode* cn) {
    CompositeConcatNode nodes = UnpackCompositeConcat(cn);
    ICHECK(nodes.concat != nullptr)
        << "Marvell-Compiler-ERROR-Internal::attribute concat can't be nullptr";

    std::string mrvlLayerName = "Concat";
    std::string name = "concat";
    std::string data_layout;
    std::string out_layout;
    std::vector<JSONGraphNodeEntry> inputs;

    for (auto arg : cn->args) {
      inputs.push_back(VisitExpr(arg)[0]);
    }

    std::vector<int64_t> layout_vec;
    GetInputTensorShapeViaArgN(cn, &layout_vec);
    if (layout_vec.size() == 4) {
      data_layout = "NHWC";
      out_layout = "NHWC";
    } else if (layout_vec.size() == 2) {
      data_layout = "NC";
      out_layout = "NC";
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.concat);
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, "", out_layout);

    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite fc (fully-connected) operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlFcLayer(const CallNode* cn) {
    CompositeFcNode nodes = UnpackCompositeFc(cn);

    std::string name = "nn.fc_ni2no";
    std::string mrvlLayerName = "FC";
    std::string data_layout = "NC";
    std::string kernel_layout = "OI";
    std::string out_layout = "NC";
    std::string bias_layout = "-O";
    std::vector<JSONGraphNodeEntry> inputs;

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.fc->args[1])[0]);
    if (nodes.add) {
      inputs.push_back(VisitExpr(nodes.add->args[1])[0]);
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    std::vector<std::string> kernel_const_name = {layer_name_ + "_const_" +
                                                  std::to_string(const_suffix_++)};
    JsonNodeSetAttr(json_node, "kernel_const_name", kernel_const_name);
    SetCallNodeAttribute(json_node, nodes.fc);
    if (nodes.add) {
      SetCallNodeAttribute(json_node, nodes.add);
      std::vector<std::string> bias_const_name = {layer_name_ + "_const_" +
                                                  std::to_string(const_suffix_++)};
      JsonNodeSetAttr(json_node, "bias_const_name", bias_const_name);
      JsonNodeSetAttr(json_node, "bias_layout", {bias_layout});
    }
    if (nodes.activation) JsonNodeSetAttr(json_node, "activation_type", {"relu"});
    if (nodes.transform && nodes.flatten) {
      JsonNodeSetAttr(json_node, "weights_need_transform", {"yes"});
      data_layout = "NHWC";
    }
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, kernel_layout,
                            out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite (global) maxpooling operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlMaxpool2DLayer(const CallNode* cn) {
    std::string mrvlLayerName = "Maxpool2D";
    CompositePoolNode nodes = UnpackCompositePool(cn, mrvlLayerName);
    const auto* maxpool_attr = nodes.pool->attrs.as<MaxPool2DAttrs>();
    std::string name = "nn.maxpool2d_nhwc2nhwc";
    std::string data_layout = maxpool_attr->layout;
    std::string out_layout = maxpool_attr->layout;
    std::vector<JSONGraphNodeEntry> inputs;

    ICHECK(maxpool_attr) << "Marvell-Compiler-ERROR-Internal::Downcast to MaxPool2DAttrs failed.";
    ICHECK(maxpool_attr->layout == "NHWC")
        << "Marvell-Compiler-ERROR-Internal::"
        << "Layout must be NHWC, has the module been pre-processed correctly?";

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.pool);
    auto pool_attrs = nodes.pool->attrs.as<MaxPool2DAttrs>();
    std::vector<int64_t> kernel_layout_vec;
    kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[0])));
    kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[1])));
    JsonNodeSetVecAttr(json_node, "kernel_layout_shape", kernel_layout_vec);
    if (nodes.pad) SetMrvlLayerPadAttrs(json_node, nodes.pad);
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, "HW",
                            out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite (global) avgpooling operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlAvgpool2DLayer(const CallNode* cn) {
    std::string mrvlLayerName = "Avgpool2D";
    CompositePoolNode nodes = UnpackCompositePool(cn, mrvlLayerName);
    const auto* avgpool_attr = nodes.pool->attrs.as<AvgPool2DAttrs>();
    std::string name = "nn.avgpool2d_nhwc2nhwc";
    std::string data_layout = avgpool_attr->layout;
    std::string out_layout = avgpool_attr->layout;
    std::vector<JSONGraphNodeEntry> inputs;

    ICHECK(avgpool_attr) << "Marvell-Compiler-ERROR-Internal::Downcast to AvgPool2DAttrs failed.";
    ICHECK(avgpool_attr->layout == "NHWC")
        << "Marvell-Compiler-ERROR-Internal::"
        << "Layout must be NHWC, has the module been pre-processed correctly?";

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.pool);
    auto pool_attrs = nodes.pool->attrs.as<AvgPool2DAttrs>();
    std::vector<int64_t> kernel_layout_vec;
    kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[0])));
    kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[1])));
    JsonNodeSetVecAttr(json_node, "kernel_layout_shape", kernel_layout_vec);
    if (nodes.pad) SetMrvlLayerPadAttrs(json_node, nodes.pad);
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, "HW",
                            out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite globalavgpooling operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlGlobalAvgpool2DLayer(const CallNode* cn) {
    std::string mrvlLayerName = "GlobalAvgpool2D";
    CompositePoolNode nodes = UnpackCompositePool(cn, mrvlLayerName);
    const auto* globalavgpool_attr = nodes.pool->attrs.as<GlobalPool2DAttrs>();
    std::string name = "nn.globalavgpool2d_nhwc2nhwc";
    std::string data_layout = globalavgpool_attr->layout;
    std::string out_layout = globalavgpool_attr->layout;
    std::vector<JSONGraphNodeEntry> inputs;

    ICHECK(globalavgpool_attr)
        << "Marvell-Compiler-ERROR-Internal::Downcast to GlobalPool2DAttrs failed.";
    ICHECK(globalavgpool_attr->layout == "NHWC")
        << "Marvell-Compiler-ERROR-Internal::"
        << "Layout must be NHWC, has the module been pre-processed correctly?";

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    std::vector<int64_t> kernel_layout_vec;
    std::vector<int64_t> data_layout_vec;
    GetInputTensorShapeViaArgN(cn, &data_layout_vec);
    ICHECK(data_layout_vec.size() == 4);
    kernel_layout_vec.push_back(data_layout_vec[1]);
    kernel_layout_vec.push_back(data_layout_vec[2]);
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.pool);
    JsonNodeSetVecAttr(json_node, "kernel_layout_shape", kernel_layout_vec);
    if (nodes.pad) SetMrvlLayerPadAttrs(json_node, nodes.pad);

    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, "HW",
                            out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite globalmaxpooling operator.
   *
   * A composite function is only created when using the uint8 datatype for these operators.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlGlobalMaxpool2DLayer(const CallNode* cn) {
    std::string mrvlLayerName = "GlobalMaxpool2D";
    std::string name = "nn.globalmaxpool2d_nhwc2nhwc";
    CompositePoolNode nodes = UnpackCompositePool(cn, mrvlLayerName);

    const auto* globalmaxpool_attr = nodes.pool->attrs.as<GlobalPool2DAttrs>();
    ICHECK(globalmaxpool_attr)
        << "Marvell-Compiler-ERROR-Internal::Downcast to GlobalPool2DAttrs failed.";
    ICHECK(globalmaxpool_attr->layout == "NHWC")
        << "Marvell-Compiler-ERROR-Internal::"
        << "Layout must be NHWC, has the module been pre-processed correctly?";

    std::string data_layout = globalmaxpool_attr->layout;
    std::string out_layout = globalmaxpool_attr->layout;
    std::vector<JSONGraphNodeEntry> inputs;
    std::vector<int64_t> kernel_layout_vec;
    std::vector<int64_t> data_layout_vec;
    GetInputTensorShapeViaArgN(cn, &data_layout_vec);
    ICHECK(data_layout_vec.size() == 4);
    kernel_layout_vec.push_back(data_layout_vec[1]);
    kernel_layout_vec.push_back(data_layout_vec[2]);
    inputs.push_back(VisitExpr(cn->args[0])[0]);

    // op_type_ is "kernel"
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.pool);
    JsonNodeSetVecAttr(json_node, "kernel_layout_shape", kernel_layout_vec);
    if (nodes.pad) SetMrvlLayerPadAttrs(json_node, nodes.pad);

    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout, "HW",
                            out_layout);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of an OpNode layer.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateMrvlLayer4OpNode(const CallNode* cn) {
    const auto* op_node = cn->op.as<OpNode>();
    ICHECK(op_node) << "Marvell-Compiler-ERROR-Internal::Downcast to OpNode failed.";
    String op_name = op_node->name;

    std::string name = op_name;
    std::string mrvlLayerName = op_name;
    std::string data_layout = "";
    std::string out_layout = "";
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    // op_type_ is "kernel"
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    if (op_name == "transpose") {
      SetCallNodeAttribute(json_node, cn);
    } else if (op_name == "layout_transform") {
      SetCallNodeAttribute(json_node, cn);
      auto layout_transform_attr = cn->attrs.as<LayoutTransformAttrs>();
      data_layout = layout_transform_attr->src_layout;
      out_layout = layout_transform_attr->dst_layout;
    } else {
      LOG(FATAL) << "Can't handle this OpNode: " << AsText(GetRef<Call>(cn), false);
    }
    SetMrvlLayerCommonAttrs(json_node, cn, layer_name_, mrvlLayerName, data_layout,
                            "" /* no kernel_layout */, out_layout);
    return json_node;
  }
};

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}

/*!
 * \brief Generate compiled model binary and then return a runtime module for Mrvl.
 *
 * \note This consists of a series of IR functions, which each represents
 * a full Mrvl subgraph/region (in tvmc mode) or one fused Mrvl backend layer
 * macro function (in dbg mode), that they can be computed on Mrvl accelerator.
 *
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module MrvlCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>())
      << "Marvell-Compiler-ERROR-Internal::Downcast to FunctionNode failed.";

  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);
  const std::string mrvl_run_mode = func->GetAttr<String>("mode").value();
  runtime::Module runtime_lib;

  // Extract attributes from the frontend to be passed to the runtime
  const std::string compiler_opt = func->GetAttr<String>("compiler_opts_string").value();
  MrvlJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();

  // Collect Nodes.json and Const.json
  const auto* get_json = runtime::Registry::Get("tvm.mrvl.GetNodesJSONString");
  std::string nodes_json_string = (*get_json)(graph_json);
  auto consts_json_string = serializer.GetConstJSONString();

  // Rename constants to a form acceptable by backend
  const auto* modifyConsts = runtime::Registry::Get("tvm.mrvl.ModifyConstNames");
  std::string modified_json = (*modifyConsts)(nodes_json_string, consts_json_string);
  auto json_vec = split(modified_json, '|');

  // Extract attributes from the nodes_json by key-value lookup using Python API
  // These are passed to hardware runtime module for initialization
  const tvm::runtime::PackedFunc* json_lookup;
  json_lookup = runtime::Registry::Get("tvm.mrvl.find_value_in_KV_pair");
  const std::string string_inp = (*json_lookup)(nodes_json_string, "num_subgraph_inputs");
  const int num_inputs = std::stoi(string_inp);
  const std::string string_out = (*json_lookup)(nodes_json_string, "num_subgraph_outputs");
  const int num_outputs = std::stoi(string_out);
  const std::string string_bsize = (*json_lookup)(nodes_json_string, "batch_size");
  const int batch_size = std::stoi(string_bsize);

  // Invoke Marvell Backend compiler to generate binary for sub graph
  const auto* compile = runtime::Registry::Get("tvm.mrvl.CompileModel");
  std::string bin = (*compile)(func_name, json_vec[0], json_vec[1], compiler_opt);

  if (mrvl_run_mode == "sim") {
    const auto* pf = runtime::Registry::Get("runtime.mrvl_runtime_create");
    ICHECK(pf != nullptr) << "Cannot find software simulator runtime module to create";
    runtime_lib = (*pf)(func_name, json_vec[0], bin);
  } else if (mrvl_run_mode == "hw") {
    const auto* pf = runtime::Registry::Get("runtime.mrvl_hw_runtime_create");
    ICHECK(pf != nullptr) << "Cannot find hardware runtime module to create";
    runtime_lib = (*pf)(func_name, json_vec[0], bin, num_inputs, num_outputs, batch_size);
  } else {
    ICHECK(0) << "Unrecognized Marvell Run Mode! " << mrvl_run_mode;
  }

  return runtime_lib;
}

TVM_REGISTER_GLOBAL("relay.ext.mrvl").set_body_typed(MrvlCompiler);

}  // namespace mrvl
}  // namespace contrib

}  // namespace relay
}  // namespace tvm
