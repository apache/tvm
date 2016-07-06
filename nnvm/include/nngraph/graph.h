/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Configuation of nngraph as well as basic data structure.
 */
#ifndef NNGRAPH_GRAPH_H_
#define NNGRAPH_GRAPH_H_

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "./base.h"
#include "./node.h"

namespace nngraph {

/*!
 * \brief Symbolic computation graph.
 */
class Graph {
 public:
  /*! \brief outputs of the computation graph. */
  std::vector<NodeEntry> outputs;
  /*!
   * \brief attributes of a graph
   *  Each attribute is immutable,
   *  and can be shared across multiple Instance of graph
   */
  std::unordered_map<std::string, std::shared_ptr<const any> > attrs;
  /*!
   * \brief perform a Post Order DFS visit to each node in the graph.
   *  This order is deterministic and is also topoligical sorted.
   * \param fvisit a function of type std::function<void(const std::shared_ptr<Node>&)>
   * \tparam FVisit The function type to perform the visit.
   */
  template<typename FVisit>
  inline void DFSVisit(FVisit fvisit) const;
};

// inline function implementations
template <typename GNode, typename HashType,
           typename FVisit, typename HashFunc,
          typename InDegree, typename GetInput>
void PostOrderDFSVisit(const std::vector<GNode>& heads,
                       FVisit fvisit,
                       HashFunc hash,
                       InDegree indegree,
                       GetInput getinput) {
  std::vector<std::pair<GNode, uint32_t> > stack;
  std::unordered_set<HashType> visited;
  for (auto& head : heads) {
    HashType head_hash = hash(head);
    if (visited.count(head_hash) == 0) {
      stack.push_back(std::make_pair(head, 0));
      visited.insert(head_hash);
    }
    while (!stack.empty()) {
      std::pair<GNode, uint32_t>& back = stack.back();
      if (back.second == indegree(back.first)) {
        fvisit(back.first);
        stack.pop_back();
      } else {
        const GNode& input = getinput(back.first, back.second++);
        HashType input_hash = hash(input);
        if (visited.count(input_hash) == 0) {
          stack.push_back(std::make_pair(input, 0));
          visited.insert(input_hash);
        }
      }
    }
  }
}

template<typename FVisit>
inline void Graph::DFSVisit(FVisit fvisit) const {
  typedef const std::shared_ptr<Node>* GNode;
  std::vector<GNode> head_nodes(outputs.size());
  std::transform(outputs.begin(), outputs.end(), head_nodes.begin(),
                 [](const NodeEntry& e)->GNode {
                   return &e.node;
                 });
  PostOrderDFSVisit<GNode, Node*>(
      head_nodes,
      [fvisit](GNode n) { fvisit(*n); },  // FVisit
      [](GNode n)->Node* { return n->get(); },  // HashFunc
      [](GNode n)->uint32_t {  // InDegree
        return (*n)->inputs.size() + (*n)->control_deps.size();
      },
      [](GNode n, uint32_t index)->GNode {  // GetInput
        if (index < (*n)->inputs.size()) {
          return &(*n)->inputs.at(index).node;
        } else {
          return &(*n)->control_deps.at(index - (*n)->inputs.size());
        }
      });
}

}  // namespace nngraph

#endif  // NNGRAPH_GRAPH_H_
