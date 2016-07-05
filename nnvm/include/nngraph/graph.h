/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Configuation of nngraph as well as basic data structure.
 */
#ifndef NNGRAPH_GRAPH_H_
#define NNGRAPH_GRAPH_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "./node.h"

namespace nngraph {

/*!
 * \brief Symbolic computation graph.
 */
class Graph {
 public:
  /*! \brief outputs of the computation graph. */
  std::vector<NodeEntry> outputs;
  /*! \brief attributes of a graph */
  std::unordered_map<std::string, any> attrs;
};

}  // namespace nngraph

#endif  // NNGRAPH_GRAPH_H_
