/*!
 *  Copyright (c) 2018 by Contributors
 * \file reorg.h
 */
#ifndef NNVM_TOP_VISION_YOLO_REORG_H_
#define NNVM_TOP_VISION_YOLO_REORG_H_

#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <sstream>

namespace nnvm {
namespace top {

template <typename AttrType,
          bool (*is_none)(const AttrType &),
          bool (*assign)(AttrType *,
          const AttrType &),
          bool reverse_infer,
          std::string (*attr_string)(const AttrType &),
          int n_in = -1,
          int n_out = -1>
inline bool ReorgAttr(const nnvm::NodeAttrs &attrs,
                      std::vector<AttrType> *in_attrs,
                      std::vector<AttrType> *out_attrs,
                      const AttrType &none) {
  AttrType dattr = none;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  if (n_in != -1) {
    in_size = static_cast<size_t>(n_in);
  }
  if (n_out != -1) {
    out_size = static_cast<size_t>(n_out);
  }

  auto deduce = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
    for (size_t i = 0; i < size; ++i) {
      if (i == 0) {
        CHECK(assign(&dattr, (*vec)[i]))
            << "Incompatible attr in node " << attrs.name << " at " << i
            << "-th " << name << ": "
            << "expected " << attr_string(dattr) << ", got "
            << attr_string((*vec)[i]);
      }
    }
  };
  deduce(in_attrs, in_size, "input");

  auto write = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
    for (size_t i = 0; i < size; ++i) {
      CHECK(assign(&(*vec)[i], dattr))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": "
          << "expected " << attr_string(dattr) << ", got "
          << attr_string((*vec)[i]);
    }
  };
  write(out_attrs, out_size, "output");

  if (is_none(dattr)) {
    return false;
  }
  return true;
}

template <int n_in, int n_out>
inline bool ReorgShape(const NodeAttrs &attrs,
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in))
        << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out))
        << " in operator " << attrs.name;
  }
  return ReorgAttr<TShape, shape_is_none, shape_assign, true, shape_string>(
      attrs, in_attrs, out_attrs, TShape());
}

template <int n_in, int n_out>
inline bool ReorgType(const NodeAttrs &attrs,
                      std::vector<int> *in_attrs,
                      std::vector<int> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in))
        << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out))
        << " in operator " << attrs.name;
  }
  return ReorgAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_attrs, out_attrs, -1);
}

struct ReorgParam : public dmlc::Parameter<ReorgParam> {
  int stride;

  DMLC_DECLARE_PARAMETER(ReorgParam) {
    DMLC_DECLARE_FIELD(stride).set_default(1).describe("Stride value");
  }
};
}  // namespace top
}  // namespace nnvm
#endif  // NNVM_TOP_VISION_YOLO_REORG_H_
