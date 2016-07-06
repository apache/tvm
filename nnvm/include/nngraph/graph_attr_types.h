/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph_attr_types.h
 * \brief Data structures that can appear in graph attributes.
 */
#ifndef NNGRAPH_GRAPH_ATTR_TYPES_H_
#define NNGRAPH_GRAPH_ATTR_TYPES_H_

#include <vector>
#include <unordered_map>
#include "./graph.h"

namespace nngraph {

/*!
 * \brief Auxililary data structure to index a graph.
 *  It maps Nodes in the graph to consecutive integers node_id.
 *  It also maps IndexedGraph::NodeEntry to consecutive integer entry_id.
 *  This allows storing properties of Node and NodeEntry into
 *  compact vector and quickly access them without resorting to hashmap.
 */
struct IndexedGraph {
 public:
  /*! \brief represents a data in the graph */
  struct NodeEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t node_id;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*!
     * \brief compare equality
     * \param other the other entry to compare
     * \return whether two entries equals to each other
     */
    inline bool operator==(const NodeEntry& other) const {
      return node_id == other.node_id && index == other.index;
    }
  };
  /*! \brief Node data structure in IndexedGraph */
  struct Node {
    /*! \brief pointer to the source node */
    const nngraph::Node* source;
    /*! \brief inputs to the node */
    array_view<NodeEntry> inputs;
    /*! \brief control flow dependencies to the node */
    array_view<uint32_t> control_deps;
  };
  /*! \return number of nodes in the graph */
  inline size_t num_nodes() const {
    return nodes_.size();
  }
  /*! \return total number of NodeEntry in the graph */
  inline size_t num_node_entries() const {
    return entry_rptr_.back();
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given IndexedGraph::NodeEntry
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const NodeEntry& e) const {
    return entry_rptr_[e.node_id] + e.index;
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given NodeEntry.
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const nngraph::NodeEntry& e) const {
    return entry_rptr_[node_id(e.node.get())] + e.index;
  }
  /*!
   * \brief Get the corresponding node id for a given Node in the IndexedGraph.
   * \param node The Node to query for index.
   * \return the node index.
   */
  inline uint32_t node_id(const nngraph::Node* node) const {
    return node2index_.at(node);
  }
  /*!
   * \brief Get the corresponding Node structure for a given node_id.
   * \param node_id The node id
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](uint32_t node_id) const {
    return nodes_[node_id];
  }
  /*!
   * \brief Get the corresponding Node structure
   * \param node The pointer to the Node structure
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](const nngraph::Node* node) const {
    return nodes_[node_id(node)];
  }
  /*! \return list of argument nodes */
  inline const std::vector<uint32_t>& arg_nodes() const {
    return arg_nodes_;
  }
  /*!
   * \brief Constructor an IndexedGraph from normal Graph
   * \param other The source graph.
   */
  explicit IndexedGraph(const Graph& other);
  // disallow copy assign
  IndexedGraph(const IndexedGraph& other) = delete;

 private:
  // node pointers in CSR structure.
  std::vector<Node> nodes_;
  // index to argument nodes
  std::vector<uint32_t> arg_nodes_;
  // mapping from node to index.
  std::unordered_map<const nngraph::Node*, uint32_t> node2index_;
  // CSR pointer of node entries
  std::vector<size_t> entry_rptr_;
  // space to store input entries of each
  std::vector<NodeEntry> input_entries_;
  // control flow dependencies
  std::vector<uint32_t> control_deps_;
};

}  // namespace nngraph

#endif  // NNGRAPH_GRAPH_ATTR_TYPES_H_
