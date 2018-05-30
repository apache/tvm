/*!
 *  Copyright (c) 2017 by Contributors
 * \file common_sub_expression.cc
 * \brief take common expressions out and replace with one.
 *
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>

namespace nnvm {
namespace compiler {

nnvm::Graph CommonSubExpression(nnvm::Graph src) {
  std::vector<NodeEntry> commonNodeList;
  std::unordered_map<std::string, int> unique_expressn;
  std::string expressn = "";
  std::function<std::string(const NodeEntry& node, std::string)>find_cummulative_expression
    = [&find_cummulative_expression, &commonNodeList, &unique_expressn]
    (const NodeEntry& node, std::string expressn)->std::string {
    std::string tempExpressn = "";
    // If varibale, just get the attribute name
    if (node.node->is_variable()) {
      tempExpressn += node.node->attrs.name;
    } else {
      for (auto& e : node.node->inputs) {
        tempExpressn = find_cummulative_expression(e, tempExpressn);
        if (unique_expressn.count(tempExpressn)) {
          // Replace current one with already commoned node
          e = commonNodeList[unique_expressn.at(tempExpressn)];
          // Replace already commoned expression with its global index
          tempExpressn = std::to_string(unique_expressn.at(tempExpressn));
        }
        expressn += tempExpressn;
        tempExpressn = "";
      }

      // Append all the params name & value also in the expressn
      for (auto kv : node.node->attrs.dict) {
        expressn = kv.first + kv.second + expressn;
      }
      tempExpressn = node.node->op()->name + expressn;
    }
    return tempExpressn;
  };

  DFSVisit(src.outputs, [&](const nnvm::NodePtr& n) {
    // If variable then, there is nothing to take common
    if (!n->is_variable()) {
      // If non-variable, then form logical expression
      expressn = "";
      expressn = find_cummulative_expression(nnvm::NodeEntry{n, 0, 0}, expressn);
      commonNodeList.push_back(nnvm::NodeEntry{n, 0, 0});
      unique_expressn.emplace(expressn, commonNodeList.size() - 1);
    }
  });

  return src;
}

NNVM_REGISTER_PASS(CommonSubExpression)
.set_body(CommonSubExpression);
}  // namespace compiler
}  // namespace nnvm
