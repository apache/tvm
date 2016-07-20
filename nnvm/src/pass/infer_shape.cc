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

Graph InferShape(Graph ret) {
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape = Op::GetAttr<FInferShape>("FInferShape");
  // reshape shape vector
  ShapeVector rshape(idx.num_node_entries());

  if (ret.attrs.count("shape_args") != 0) {
    const ShapeVector& shape_args = ret.GetAttr<ShapeVector>("shape_args");
    CHECK_LE(shape_args.size(), idx.arg_nodes().size())
        << "shape args is more than number of arguments";
    for (size_t i = 0; i < shape_args.size(); ++i) {
      rshape[idx.entry_id(idx.arg_nodes()[i], 0)] = shape_args[i];
    }
  }
  std::string shape_attr_key;
  if (ret.attrs.count("shape_attr_key") != 0) {
    shape_attr_key = ret.GetAttr<std::string>("shape_attr_key");
  }

  // temp space for shape inference.
  std::vector<TShape*> ishape, oshape;
  // number of completed nodes
  size_t num_known = 0;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      if (shape_attr_key.length() != 0) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          CHECK_EQ(inode.source->num_outputs(), 1);
          std::istringstream is(it->second);
          CHECK(is >> rshape[idx.entry_id(nid, 0)]) << "Invalid shape attribute";
        }
      }
      continue;
    }
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

NNVM_REGISTER_PASS(InferShape)
.describe("Infer the shape of each node entries.")
.set_body(InferShape)
.set_change_graph(false)
.provide_graph_attr("shape");

DMLC_JSON_ENABLE_ANY(ShapeVector, list_shape);

}  // namespace pass
}  // namespace nnvm
