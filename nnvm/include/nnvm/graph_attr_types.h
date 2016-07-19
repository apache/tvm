/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph_attr_types.h
 * \brief Data structures that can appear in graph attributes.
 */
#ifndef NNVM_GRAPH_ATTR_TYPES_H_
#define NNVM_GRAPH_ATTR_TYPES_H_

#include <vector>
#include <string>
#include "./tuple.h"

namespace nnvm {

/*!
 * \brief The result holder of JSON serializer
 *
 * \note Stored under ret.attrs["json"], provided by Pass "SaveJSON"

 * \code
 *  Graph ret = ApplyPass(src_graph, {"SaveJSON"});
 *  const JSONString& json = ret.GetAttr<JSONString>("shape");
 * \endcode
 */
using JSONString = std::string;

/*!
 * \brief The result holder of shape of each NodeEntry in the graph.
 * \note Stored under graph.attrs["shape"], provided by Pass "InferShape"
 *
 * \code
 *  Graph g = ApplyPass(src_graph, {"InferShape"});
 *  const ShapeVector& shapes = g.GetAttr<ShapeVector>("shape");
 *  // get shape by entry id
 *  TShape entry_shape = shapes[g.indexed_graph().entry_id(my_entry)];
 * \endcode
 *
 * \sa FInferShape
 */
using ShapeVector = std::vector<TShape>;

}  // namespace nnvm

#endif  // NNVM_GRAPH_ATTR_TYPES_H_
