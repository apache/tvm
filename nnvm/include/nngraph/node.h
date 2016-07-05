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
  /*!
   * \brief The operator this node uses.
   *  For place holder variable, op == nullptr.
   */
  const Op *op;
  /*! \brief inputs to this node */
  std::vector<NodeEntry> inputs;
  /*!
   * \brief Optional control flow dependencies
   *  Gives operation must be performed before this operation.
   */
  std::vector<std::shared_ptr<Node> > control_deps;
  /*! \brief The attributes in the node. */
  NodeAttrs attrs;
  /*! \brief destructor of node */
  ~Node();
  /*!
   * \brief return whether node is placeholder variable.
   *  This is equivalent to op == nullptr
   * \return whether node is placeholder input variable
   */
  inline bool is_variable() const;
  /*!
   * \brief create a new empty shared_ptr of Node.
   * \return a created empty node.
   */
  static std::shared_ptr<Node> Create();
};

// implementation of functions.
inline bool Node::is_variable() const {
  return this->op == nullptr;
}

}  // namespace nngraph

#endif  // NNGRAPH_NODE_H_
