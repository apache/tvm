/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_shape.cc
 * \brief Inference the shapes given existin information.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

template<typename AttrType, typename IsNone>
Graph InferAttr(Graph &&ret,
                const AttrType def_value,
                const char* infer_name,
                const char* input_name,
                const char* attr_key_name,
                const char* attr_name,
                const char* unknown_name,
                IsNone fis_none) {
  using AttrVector = std::vector<AttrType>;
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape =
      Op::GetAttr<FInferNodeEntryAttr<AttrType> >(infer_name);
  static auto& is_backward =
      Op::GetAttr<TIsBackwardOp>("TIsBackwardOp");
  // reshape shape vector
  AttrVector rshape(idx.num_node_entries(), def_value);

  if (ret.attrs.count(input_name) != 0) {
    const AttrVector& shape_args = ret.GetAttr<AttrVector>(input_name);
    CHECK_LE(shape_args.size(), idx.input_nodes().size())
        << "shape args is more than number of arguments";
    for (size_t i = 0; i < shape_args.size(); ++i) {
      rshape[idx.entry_id(idx.input_nodes()[i], 0)] = shape_args[i];
    }
    // erase the provided arguments
    ret.attrs.erase(input_name);
  }
  std::string shape_attr_key;
  if (ret.attrs.count(attr_key_name) != 0) {
    shape_attr_key = ret.GetAttr<std::string>(attr_key_name);
    // erase the provided arguments
    ret.attrs.erase(attr_key_name);
  }

  // temp space for shape inference.
  std::vector<AttrType*> ishape, oshape;
  // number of completed nodes
  size_t num_unknown = 0;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) {
      if (shape_attr_key.length() != 0 && fis_none(rshape[idx.entry_id(nid, 0)])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          CHECK_EQ(inode.source->num_outputs(), 1);
          std::istringstream is(it->second);
          CHECK(is >> rshape[idx.entry_id(nid, 0)]) << "Invalid attribute";
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
      num_unknown +=
          !(finfer_shape[inode.source->op](inode.source->attrs, ishape, oshape));
    } else if (is_backward.get(inode.source->op, false)) {
      // backward operator inference.
      CHECK_GE(inode.control_deps.size(), 1)
          << "BackwardOp need to have control_deps to its forward op";
      const auto& fnode = idx[inode.control_deps[0]];
      CHECK_EQ(fnode.inputs.size(), inode.source->num_outputs())
          << "BackwardOp need to correspond to the forward node";
      bool known = true;
      for (size_t i = 0; i < fnode.inputs.size(); ++i) {
        *oshape[i] = rshape[idx.entry_id(fnode.inputs[i])];
        if (fis_none(*oshape[i])) known = false;
      }
      num_unknown += !known;
    }
  }
  // set the shapes
  ret.attrs[attr_name] = std::make_shared<any>(std::move(rshape));
  // number of nodes who knows the shape.
  ret.attrs[unknown_name] = std::make_shared<any>(num_unknown);
  return ret;
}

NNVM_REGISTER_PASS(InferShape)
.describe("Infer the shape of each node entries.")
.set_body([](Graph ret) {
    return InferAttr<TShape>(
        std::move(ret), TShape(),
        "FInferShape", "shape_args", "shape_attr_key",
        "shape", "shape_num_unknown_nodes",
        [](const TShape& s) { return s.ndim() == 0; });
  })
.set_change_graph(false)
.provide_graph_attr("shape");

NNVM_REGISTER_PASS(InferType)
.describe("Infer the dtype of each node entries.")
.set_body([](Graph ret) {
    return InferAttr<int>(
        std::move(ret), 0,
        "FInferType", "dtype_inputs", "dtype_attr_key",
        "dtype", "dtype_num_unknown_nodes",
        [](const int t) { return t == -1; });
  })
.set_change_graph(false)
.provide_graph_attr("dtype");

DMLC_JSON_ENABLE_ANY(ShapeVector, list_shape);
DMLC_JSON_ENABLE_ANY(DTypeVector, list_int);
DMLC_JSON_ENABLE_ANY(size_t, size_t);

}  // namespace
}  // namespace pass
}  // namespace nnvm
