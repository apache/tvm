/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise_op_common.h
 * \brief Common operator utilities
 */
#ifndef NNVM_TOP_ELEMWISE_OP_COMMON_H_
#define NNVM_TOP_ELEMWISE_OP_COMMON_H_

#include <string>
#include <vector>
#include <utility>
#include "./op_common.h"

namespace nnvm {
namespace top {

template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&),
         int n_in = -1, int n_out = -1>
inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs,
                         const AttrType& none) {
  AttrType dattr = none;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  if (n_in != -1)
    in_size = static_cast<size_t>(n_in);
  if (n_out != -1)
    out_size = static_cast<size_t>(n_out);

  auto deduce = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&dattr, (*vec)[i]))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  deduce(in_attrs, in_size, "input");
  if (reverse_infer) deduce(out_attrs, out_size, "output");

  auto write = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&(*vec)[i], dattr))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  write(in_attrs, in_size, "input");
  write(out_attrs, out_size, "output");

  if (is_none(dattr)) return false;
  return true;
}

template<int n_in, int n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, TShape());
}

template<int n_in, int n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

#define NNVM_REGISTER_ELEMWISE_UNARY_OP(name)                       \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "Tensor", "The input tensor.")
}  // namespace top
}  // namespace nnvm
#endif  // NNVM_TOP_ELEMWISE_OP_COMMON_H_
