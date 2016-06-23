/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Configuation of nngraph as well as basic data structure.
 */
#ifndef NNGRAPH_GRAPH_H_
#define NNGRAPH_GRAPH_H_

#include <vector>
#include "./node.h"
#include "./attr_frame.h"

namespace nngraph {
/*!
 * \brief Symbolic computation graph.
 */
class Graph {
 public:
  /*!
   * \brief get the index th element from the returned tuple.
   * \param index index of multi output
   * \return the symbol corresponds to the indexed element.
   */
  Graph operator[] (size_t index) const;
  /*!
   * \brief get number of outputs of this symbol
   * \return number of outputs
   */
  inline size_t outputs_size() const {
    return outputs_.size();
  }

 private:
  /*! \brief outputs of the graph. */
  std::vector<NodeEntry> outputs_;
  /*! \brief additional internal attribute */
  AttrFrame attr_frame_;
};

}  // namespace nngraph

#endif  // NNGRAPH_GRAPH_H_
