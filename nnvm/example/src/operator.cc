// Copyright (c) 2016 by Contributors
// This is an example on how we can register operator information to NNVM
// these operator information are used to support various graph building and optimizations
// see tests/python/ folder for the test-cases that uses these information.

#include <nnvm/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/node.h>
#include <nnvm/graph_attr_types.h>
#include <utility>

namespace myproject {

using nnvm::FListInputNames;
using nnvm::FMutateInputs;
using nnvm::FInferShape;
using nnvm::FInferType;
using nnvm::FInplaceOption;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::FGradient;
using nnvm::NodeAttrs;
using nnvm::TShape;
using nnvm::array_view;

// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  if (ishape->size() == 0 || (*ishape)[0].ndim() == 0) return false;
  for (TShape& pshape : *oshape) {
    pshape = (*ishape)[0];
  }
  for (TShape& pshape : *ishape) {
    pshape = (*ishape)[0];
  }
  return true;
}

inline std::vector<std::pair<int, int> > InplaceIn0Out0(const NodeAttrs& attrs) {
  return {{0, 0}};
}

// quick helper to make node
inline NodeEntry MakeNode(const char* op_name,
                          std::string node_name,
                          std::vector<NodeEntry> inputs) {
  NodePtr p = Node::Create();
  p->attrs.op = nnvm::Op::Get(op_name);
  p->attrs.name = std::move(node_name);
  p->inputs = std::move(inputs);
  return NodeEntry{p, 0, 0};
}

// simple demonstration of reshape.
NNVM_REGISTER_OP(reshape)
.describe("reshape source to target shape")
.set_num_inputs(1)
.set_attr_parser(
    [](NodeAttrs* attrs) {
      // parse attr parser to get target attribute
      TShape target;
      std::istringstream is(attrs->dict.at("target"));
      CHECK(is >> target);
      attrs->parsed = std::move(target);
    })
.set_attr<FInferShape>(
    "FInferShape", [] (const NodeAttrs& attrs,
                       std::vector<TShape> *ishape,
                       std::vector<TShape> *oshape) {
      // get parsed attribute
      const TShape& target = nnvm::get<TShape>(attrs.parsed);
      (*oshape)[0] = target;
      if ((*ishape)[0].ndim() == 0) return false;
      CHECK_EQ((*ishape)[0].Size(), target.Size())
          << "Reshape op: source target shape mismatch";
      return true;
    })
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0);


NNVM_REGISTER_OP(cast)
.describe("cast source type to target")
.set_num_inputs(1)
.set_attr_parser(
    [](NodeAttrs* attrs) {
      // parse attr parser to get target attribute
      int dtype;
      std::istringstream is(attrs->dict.at("dtype"));
      CHECK(is >> dtype);
      attrs->parsed = std::move(dtype);
    })
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInferType>(
    "FInferType", [](const NodeAttrs& attrs,
                     std::vector<int> *itype,
                     std::vector<int> *otype) {
      (*otype)[0] = nnvm::get<int>(attrs.parsed);
      return true;
    });

NNVM_REGISTER_OP(exp)
.describe("take exponential")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{
        MakeNode("mul", n->attrs.name + "_grad",
                 {ograds[0], NodeEntry{n, 0, 0}})
      };
    });

NNVM_REGISTER_OP(identity)
.describe("identity function")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds) {
      return std::vector<NodeEntry>{ograds[0]};
    });

NNVM_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2)
.add_alias("__add_symbol__")
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{ograds[0], ograds[0]};
    });

NNVM_REGISTER_OP(mul)
.describe("multiply two data together")
.set_num_inputs(2)
.set_attr<FInferShape>("FInferShape", SameShape)
.set_attr<FInplaceOption>("FInplaceOption", InplaceIn0Out0)
.set_attr<FGradient>(
    "FGradient", [](const NodePtr& n,
                    const std::vector<NodeEntry>& ograds){
      return std::vector<NodeEntry>{
        MakeNode("mul", n->attrs.name + "_grad_0",
                 {ograds[0], n->inputs[1]}),
        MakeNode("mul", n->attrs.name + "_grad_1",
                 {ograds[0], n->inputs[0]})
      };
    });

NNVM_REGISTER_OP(__ewise_sum__)
.describe("elementwise sum")
.set_num_inputs(nnvm::kVarg);

NNVM_REGISTER_OP(__zero__)
.describe("set output to zero")
.set_num_inputs(0);

NNVM_REGISTER_OP(__one__)
.describe("set output to one")
.set_num_inputs(0);

NNVM_REGISTER_OP(cross_device_copy)
.describe("Copy data across device.")
.set_num_inputs(1)
.set_attr<FInferShape>("FInferShape", SameShape);


NNVM_REGISTER_OP(conv2d)
.describe("take conv of input")
.set_num_inputs(2)
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight"};
  });

NNVM_REGISTER_OP(add)
.set_attr<std::string>("nick_name", "plus");

NNVM_REGISTER_OP(assign)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FMutateInputs>("FMutateInputs", [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>{0};
  });

}  // namespace myproject
