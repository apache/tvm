/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_shape.cc
 * \brief Inference the shapes given
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {

Graph InferShape(const Graph& src) {
  Graph ret = src;
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape = Op::GetAttr<FInferShape>("FInferShape");
  // reshape shape vector
  ShapeVector rshape(idx.num_node_entries());
  // temp space for shape inference.
  std::vector<TShape*> ishape, oshape;
  // number of completed nodes
  size_t num_known = 0;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    ishape.resize(inode.inputs.size());
    for (uint32_t i = 0; i < ishape.size(); ++i) {
      ishape[i] = &rshape[idx.entry_id(inode.inputs[i])];
    }
    oshape.resize(inode.source->num_outputs());
    for (uint32_t i = 0; i < oshape.size(); ++i) {
      oshape[i] = &rshape[idx.entry_id(nid, i)];
    }
    if (finfer_shape.count(inode.source->op)) {
      num_known +=
          finfer_shape[inode.source->op](inode.source->attrs, ishape, oshape);
    }
  }
  // set the shapes
  ret.attrs["shape"] = std::make_shared<any>(std::move(rshape));
  // number of nodes who knows the shape.
  ret.attrs["shape_num_known_nodes"] = std::make_shared<any>(num_known);
  return ret;
}

}  // namespace pass
}  // namespace nnvm
