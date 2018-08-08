/*!
 * Copyright (c) 2018 by Contributors
 * \file graph_fuse.h
 * \brief Definition of structs used by graph fusion
*/
#ifndef NNVM_COMPILER_GRAPH_FUSE_H_
#define NNVM_COMPILER_GRAPH_FUSE_H_

#include <nnvm/graph.h>
#include <vector>

#include "compile_engine.h"

namespace nnvm {
namespace compiler {

// The single fuse rule.
enum class FuseRule {
  kUknown,
  kFuseToMaster,
  kRealize
};

/*!
 * \brief Get DLDataType from dtype flag.
 *
 * \param type_flag The data type flag
 * \return corresponding DLDataType
 */
inline DLDataType GetDLType(int type_flag) {
  return tvm::Type2TVMType(GetTVMType(type_flag));
}

struct INodeEntryHash {
  size_t operator()(const IndexedGraph::NodeEntry& e) const {
    return e.node_id;
  }
};

struct INodeEntryEqual {
  size_t operator()(const IndexedGraph::NodeEntry &a,
                    const IndexedGraph::NodeEntry &b) const {
    return a.node_id == b.node_id && a.index == b.index;
  }
};

// Auxiliary data structure for representing fused op.
struct FuseEntry {
  // Subgraph of the fragment
  Graph subgraph;
  // The input map
  std::unordered_map<IndexedGraph::NodeEntry, nnvm::NodeEntry, INodeEntryHash,
                     INodeEntryEqual>
      imap;
  // Reverse map to the old input entry
  std::unordered_map<const Node *, IndexedGraph::NodeEntry> reverse_imap;
  // TVM Placeholder for inputs
  std::unordered_map<const Node *, Tensor> input_info;
  // Whether we can flatten data
  bool flatten_data;
  // The corresponding function.
  GraphFunc compiled_func;
};

// GroupVec stores the root node ids of the fused nodes.
using GroupVec = std::vector<int>;

// MasterVec stores master node ids of fused groups.
using MasterVec = std::vector<int>;

// FuseVec stores fused entries.
using FuseEntryVec = std::vector<FuseEntry>;

// PatternVec stores operator patterns.
using PatternVec = std::vector<TOpPattern>;

}  // namespace compiler
}  // namespace nnvm

#endif  // NNVM_COMPILER_GRAPH_FUSE_H_
