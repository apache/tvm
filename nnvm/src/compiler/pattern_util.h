/*!
 * Copyright (c) 2017 by Contributors
 * \file pattern_util.h
 * \brief Utilities for doing various pattern matching in graph.
*/
#ifndef NNVM_COMPILER_PATTERN_UTIL_H_
#define NNVM_COMPILER_PATTERN_UTIL_H_

#include <nnvm/graph.h>
#include <vector>
#include <utility>
#include <string>

namespace nnvm {
namespace compiler {

/*!
 * \brief find axis in oshape, such that:
 *  bias_shape = [1,1, ... oshape[axis], 1,1,]
 *
 *  This is used to detect bias or scaling factor on channel dimension.
 * \param oshape The output shape
 * \param bias_shape The shape of bias or scaling factor.
 * \return Pair of matched axis in o shape and bias_shape if found.
 */
inline std::pair<int, int> MatchBroadcast1DAxis(
    const TShape& oshape, const TShape& bias_shape) {
  dim_t axis_dim = bias_shape.ndim();
  for (dim_t i = bias_shape.ndim(); i != 0; --i, --axis_dim) {
    if (bias_shape[i - 1] != 1) break;
  }
  // everything is 1
  if (axis_dim == 0) {
    return {oshape.ndim()  - bias_shape.ndim(), 0};
  }
  axis_dim = axis_dim - 1;
  // The bias shape is not 1D
  for (dim_t i = 0; i < axis_dim; ++i) {
    if (bias_shape[i] != 1) return {-1, -1};
  }
  int axis = static_cast<int>(
      oshape.ndim() - bias_shape.ndim() + axis_dim);
  if (oshape[axis] != bias_shape[axis_dim]) return {-1, -1};
  return {axis, axis_dim};
}

/*!
 * \brief Expand bias dimension to match needed axis.
 *
 * \param bias The bias NodeEntry
 * \param out_dim output dimension.
 * \param bias_dim The current bias dimension.
 * \param axis The axis we want to match on.
 */
inline NodeEntry
ExpandBiasToMatchAxis(NodeEntry bias,
                      int out_dim,
                      int bias_dim,
                      int axis) {
  if (bias_dim != 1) {
    bias = MakeNode("squeeze", bias.node->attrs.name + "_sqz", {bias});
  }
  int num_pad_axis = out_dim - axis - 1;
  if (num_pad_axis > 0) {
    std::unordered_map<std::string, std::string> kwargs{
      {"axis", "1"},
      {"num_newaxis", std::to_string(num_pad_axis)}};
    return MakeNode("expand_dims", bias.node->attrs.name + "_expand",
                    {bias}, kwargs);

  } else {
    return bias;
  }
}

/*!
 * \brief Get the reference count of each node.
 * \param idx The IndexedGraph
 * \return ref_count vector of length number nodes.
 */
inline std::vector<uint32_t>
GetNodeRefCounts(const IndexedGraph& idx) {
  std::vector<uint32_t> ref_count(idx.num_nodes(), 0);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    for (const auto& e : inode.inputs) {
      ++ref_count[e.node_id];
    }
  }
  for (const auto& e : idx.outputs()) {
    // this line will realize all the outputs
    ref_count[e.node_id] += 1;
  }
  return ref_count;
}
}  // namespace compiler
}  // namespace nnvm
#endif  //  NNVM_COMPILER_PATTERN_UTIL_H_
