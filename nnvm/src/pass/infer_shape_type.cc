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

template<typename AttrType, typename IsNone, typename FDefault>
Graph InferAttr(Graph &&ret,
                const AttrType empty_val,
                const char* infer_name,
                const char* input_name,
                const char* attr_key_name,
                const char* attr_name,
                const char* unknown_name,
                IsNone fis_none,
                FDefault fdefault) {
  using AttrVector = std::vector<AttrType>;
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape =
      Op::GetAttr<FInferNodeEntryAttr<AttrType> >(infer_name);
  static auto& backward_map =
      Op::GetAttr<FBackwardOutToInIndex>("FBackwardOutToInIndex");
  // reshape shape vector
  AttrVector rshape;
  if (ret.attrs.count(attr_name) != 0) {
    rshape = ret.MoveCopyAttr<AttrVector>(attr_name);
  } else {
    rshape.resize(idx.num_node_entries(), empty_val);
  }

  if (ret.attrs.count(input_name) != 0) {
    const AttrVector& shape_args = ret.GetAttr<AttrVector>(input_name);
    CHECK_LE(shape_args.size(), idx.input_nodes().size())
        << "More provided shapes than number of arguments.";
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
  // Temp space for shape inference.
  std::vector<AttrType> ishape, oshape;
  size_t num_unknown;

  // inference step function for nid
  auto infer_step = [&](uint32_t nid) {
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
      if (shape_attr_key.length() != 0 && fis_none(rshape[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          std::istringstream is(it->second);
          CHECK(is >> rshape[out_ent_id]) << "Invalid attribute";
        }
      }
    } else if (backward_map.count(inode.source->op())) {
      // Backward operator inference.
      CHECK_GE(inode.control_deps.size(), 1)
          << "BackwardOp need to have control_deps to its forward op";
      const IndexedGraph::Node& fnode = idx[inode.control_deps[0]];
      // Inference the outputs of backward operator (equal to the inputs
      // of its corresponding forward operator).
      std::vector<uint32_t> out_map =
          backward_map[inode.source->op()](inode.source->attrs);
      bool known = true;
      for (size_t i = 0; i < out_map.size(); ++i) {
        uint32_t in_id = out_map[i];
        CHECK_LT(in_id, fnode.inputs.size());
        rshape[idx.entry_id(nid, i)] =
            rshape[idx.entry_id(fnode.inputs[in_id])];
        if (fis_none(rshape[idx.entry_id(nid, i)])) known = false;
      }
      num_unknown += !known;
    } else {
      bool forward_known = true;
      // Forward operator inference.
      ishape.resize(num_inputs, empty_val);
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = rshape[idx.entry_id(inode.inputs[i])];
        if (fis_none(ishape[i])) forward_known = false;
      }
      oshape.resize(num_outputs, empty_val);
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = rshape[idx.entry_id(nid, i)];
        if (fis_none(oshape[i])) forward_known = false;
      }
      if (!forward_known) {
        auto finfer = finfer_shape.get(inode.source->op(), fdefault);
        CHECK(finfer != nullptr)
          << "Attribute " << infer_name
          << " is not registed by op " << inode.source->op()->name;
        // Call inference function of the operator.
        forward_known = finfer(inode.source->attrs, &ishape, &oshape);
      }
      num_unknown += !forward_known;
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        rshape[idx.entry_id(inode.inputs[i])] = ishape[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        rshape[idx.entry_id(nid, i)] = oshape[i];
      }
    }
  };

  num_unknown = 0;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    infer_step(nid);
  }
  if (num_unknown != 0) {
    num_unknown = 0;
    // backward inference
    for (uint32_t i = idx.num_nodes(); i != 0; --i) {
      infer_step(i - 1);
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
        "FInferShape", "shape_inputs", "shape_attr_key",
        "shape", "shape_num_unknown_nodes",
        [](const TShape& s) { return s.ndim() == 0; },
        nullptr);
  })
.set_change_graph(false)
.provide_graph_attr("shape");

// inference fucntion for same type
inline bool SameType(const NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int def_v = -1;
  for (int v : *oattr) {
    if (v != -1) {
      def_v = v; break;
    }
  }
  if (def_v == -1) {
    for (int v : *iattr) {
      if (v != -1) {
        def_v = v; break;
      }
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    v = def_v;
  }
  return true;
}

NNVM_REGISTER_PASS(InferType)
.describe("Infer the dtype of each node entries.")
.set_body([](Graph ret) {
    return InferAttr<int>(
        std::move(ret), -1,
        "FInferType", "dtype_inputs", "dtype_attr_key",
        "dtype", "dtype_num_unknown_nodes",
        [](const int t) { return t == -1; },
        SameType);
  })
.set_change_graph(false)
.provide_graph_attr("dtype");

DMLC_JSON_ENABLE_ANY(ShapeVector, list_shape);
DMLC_JSON_ENABLE_ANY(DTypeVector, list_int);
DMLC_JSON_ENABLE_ANY(size_t, size_t);

}  // namespace
}  // namespace pass
}  // namespace nnvm
