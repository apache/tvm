/*!
 *  Copyright (c) 2018 by Contributors
 * \file tensorrt_subgraph_property.cc
 * \brief Graph partitioning rule for _tensorrt_subgraph_op.
 */

#include <nnvm/symbolic.h>
#include <vector>
#include <string>
#include <stack>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include "./subgraph_property.h"

namespace nnvm {
namespace pass {

std::vector<std::string> TokenizeTuple(const std::string& tuple) {
  CHECK(tuple.front() == '(' || tuple.front() == '[');
  CHECK(tuple.back() == ')' || tuple.back() == ']');
  std::stringstream ss(tuple.substr(1, tuple.size() - 2U));
  std::vector<std::string> ret;
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, ',');
    ret.push_back(substr);
  }
  CHECK(!ret.empty()) << "Tuple " << tuple << " contains no data";
  return ret;
}

bool IsTensorRTCompatibleOp(const std::unordered_set<std::string>& op_names,
                            const nnvm::Node& node) {
  if (node.is_variable() || !op_names.count(node.op()->name)) {
    return false;
  }
  const std::string& op_name = node.op()->name;
  const auto& params = node.attrs.dict;
  if (op_name == "conv2d" || op_name == "conv2d_transpose") {
    if ((params.count("layout") && params.at("layout") != "NCHW")
        || (params.count("kernel_layout") && params.at("kernel_layout") != "OIHW")
        || (params.count("out_layout") && params.at("out_layout") != "__undef__"
            && params.at("out_layout") != "NCHW")) {
      return false;
    }
  } else if (op_name == "max_pool2d" || op_name == "avg_pool2d"
             || op_name == "global_avg_pool2d" || op_name == "global_max_pool2d") {
    // only support floor mode
    if (params.count("layout") && params.at("layout") != "NCHW") {
      return false;
    }
    if (params.count("padding")) {
      const auto paddings = TokenizeTuple(params.at("padding"));
      // do not support asymmetric padding
      if (paddings.size() == 4U && (paddings[0] != paddings[2] || paddings[1] != paddings[3])) {
        return false;
      }
    }
    if (params.count("ceil_mode") &&
      (params.at("ceil_mode") == "True" || params.at("ceil_mode") == "1")) {
      return false;
    }
  } else if (op_name == "batch_norm") {
    if (params.count("axis") && params.at("axis") != "1") {
      return false;
    }
  } else if (op_name == "slice_like") {
    // only support slice on axes = (2, 3)
    if (!params.count("axis") || params.at("axis").empty()) {
      return false;
    }
    const int len = params.at("axis").size();
    CHECK_GE(len, 6);
    const std::string& axes_str = params.at("axis").substr(1, len - 2);
    size_t pos = axes_str.find(',');
    if (pos == std::string::npos) {
      return false;
    }
    if (std::stoi(axes_str.substr(0, pos)) != 2) {
      return false;
    }
    if (axes_str.substr(pos+1).find(',') != std::string::npos) {
      return false;
    }
    if (std::stoi(axes_str.substr(pos+1)) != 3) {
      return false;
    }
  }
  return true;
}

/*
 * This selects nodes for a subgraph that only contains operators
 * in a given set and it visits nodes via both input and output links.
 */
class TensorRTOpSelector: public SubgraphSelector {
 public:
  explicit TensorRTOpSelector(const std::unordered_set<std::string>& op_names)
    : op_names_(op_names) {}

  bool Select(const nnvm::Node &seed_node) {
    return !seed_node.is_variable() &&
        (IsTensorRTCompatibleOp(op_names_, seed_node) || seed_node.op()->name == "flatten");
  }

  bool SelectInput(const nnvm::Node &cur_node, const nnvm::Node &input_node) {
    return !input_node.is_variable() && (IsTensorRTCompatibleOp(op_names_, input_node)
        || (cur_node.op()->name == "dense" && input_node.op()->name == "flatten"));
  }

  bool SelectOutput(const nnvm::Node &cur_node, const nnvm::Node &output_node) {
    return !output_node.is_variable()
        && (IsTensorRTCompatibleOp(op_names_, output_node) || output_node.op()->name == "flatten");
  }

  // flatten op nodes may have been selected. It can only be placed before dense op. If not,
  // remove them from the candidates.
  // Ref: https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers
  virtual std::vector<nnvm::Node*> Filter(const std::vector<nnvm::Node*>& candidates) {
    std::unordered_set<nnvm::Node*> flatten_op_nodes;
    for (auto node : candidates) {
      CHECK(!node->is_variable());
      if (node->op()->name == "flatten") {
        CHECK(flatten_op_nodes.emplace(node).second);
      }
    }
    if (flatten_op_nodes.empty()) {
      return candidates;
    }
    for (auto node : candidates) {
      if (node->op()->name == "dense") {
        for (auto& input_entry : node->inputs) {
          if (!input_entry.node->is_variable() && input_entry.node->op()->name == "flatten") {
            flatten_op_nodes.erase(input_entry.node.get());
          }
        }
      }
    }
    // Till now, flatten_op_nodes contains the flatten op nodes that are not placed
    // before a dense op node. Remove them from the candidates.
    std::vector<nnvm::Node*> ret;
    for (auto node : candidates) {
      if (!flatten_op_nodes.count(node)) {
        ret.push_back(node);
      }
    }
    return ret;
  }

 private:
  const std::unordered_set<std::string>& op_names_;
};

/*
 * This subgraph property finds a subgraph whose nodes have only operators
 * within a set. The operators in the subgraph will be executed by _tensorrt_subgraph_op.
 */
class TensorRTSubgraphProperty: public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() { return std::make_shared<TensorRTSubgraphProperty>(); }
  nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_tensorrt_subgraph_op");
    n->attrs.name = "_tensorrt_subgraph_op" + std::to_string(subgraph_id);
    nnvm::Symbol new_sym = RemoveFlattenOpNodes(sym);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(new_sym));
    return n;
  }
  SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<TensorRTOpSelector>(
        this->GetAttr<std::unordered_set<std::string>>("op_names"));
  }

 private:
  nnvm::Symbol RemoveFlattenOpNodes(nnvm::Symbol sym) const {
    std::stack<nnvm::Node*> node_stack;
    std::unordered_set<nnvm::Node*> visited;
    for (auto& entry : sym.outputs) {
      if (!entry.node->is_variable() && !visited.count(entry.node.get())) {
        node_stack.push(entry.node.get());
        while (!node_stack.empty()) {
          nnvm::Node* cur_node = node_stack.top();
          node_stack.pop();
          visited.emplace(cur_node);
          CHECK_EQ(cur_node->is_variable(), false);
          for (auto& input_entry : cur_node->inputs) {
            while (!input_entry.node->is_variable() && input_entry.node->op()->name == "flatten") {
              CHECK_EQ(cur_node->op()->name, "dense");
              CHECK_EQ(input_entry.node->inputs.size(), 1U);
              input_entry = input_entry.node->inputs[0];
            }
            if (!input_entry.node->is_variable() && !visited.count(input_entry.node.get())) {
              node_stack.push(input_entry.node.get());
            }
          }
        }
      }
    }
    return sym;
  }
};

NNVM_REGISTER_SUBGRAPH_PROPERTY(tensorrt, TensorRTSubgraphProperty);

}  // namespace pass
}  // namespace nnvm
