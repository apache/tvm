/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Graph node data structure.
 */
#ifndef NNGRAPH_NODE_H_
#define NNGRAPH_NODE_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "./op.h"

namespace nngraph {

// Forward declare node.
class Node;

/*! \brief an entry that represents output data from a node */
struct NodeEntry {
  /*! \brief the source node of this data */
  std::shared_ptr<Node> node;
  /*! \brief index of output from the source. */
  uint32_t index;
};

/*!
 * \brief The attributes of the current operation node.
 *  Usually are additional parameters like axis,
 */
struct NodeAttrs {
  /*! \brief The dictionary representation of attributes */
  std::unordered_map<std::string, std::string> dict;
  /*!
   * \brief A parsed version of attributes,
   * This is generated if OpProperty.attr_parser is registered.
   * The object can be used to quickly access attributes.
   */
  any parsed;
};

/*!
 * \brief Node represents an operation in a computation graph.
 */
class Node {
 public:
  /*! \brief name of the node */
  std::string name;
  /*! \brief the operator this node is pointing at */
  const Op *op;
  /*! \brief inputs to this node */
  std::vector<NodeEntry> inputs;
  /*! \brief The attributes in the node. */
  NodeAttrs attrs;
  /*! \brief destructor of node */
  ~Node();
  /*!
   * \brief create a new empty shared_ptr of Node.
   * \return a created empty node.
   */
  static std::shared_ptr<Node> Create();
};

}  // namespace nngraph

#endif  // NNGRAPH_NODE_H_
