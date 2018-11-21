/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/relay/op/layout.cc
 * \brief Layout expression.
 */

#include "layout.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(LayoutNode);

std::vector<IndexExpr> ConvertLayout(
    std::vector<IndexExpr> src,
    const Layout& src_layout,
    const Layout& dst_layout) {
  CHECK_EQ(src_layout.ndim(), src.size());
  if (src_layout == dst_layout) {
    return src;
  } else if (!src_layout.defined()) {
    LOG(FATAL) << "cannot convert undefined layout to " << dst_layout;
  } else if (!dst_layout.defined()) {
    LOG(FATAL) << "cannot convert " << src_layout << " to undefined layout";
  }

  CHECK(src_layout.Convertible(dst_layout))
    << "cannot convert from "
    << src_layout << " to " << dst_layout;

  std::vector<IndexExpr> dst(dst_layout.ndim());
  for (size_t i = 0; i < src_layout.ndim(); ++i) {
    Layout::LayoutDim src_dim = src_layout[i];
    if (Layout::IsSuperdim(src_dim)) {
      int dst_major_pos = dst_layout.Indexof(Layout::ToSuperdim(src_dim));
      int dst_minor_pos = dst_layout.Indexof(Layout::ToSubdim(src_dim));
      int src_minor_pos = src_layout.Indexof(Layout::ToSubdim(src_dim));
      int src_factor = src_layout.Subsizeof(src_dim);
      int dst_factor = dst_layout.Subsizeof(src_dim);
      IndexExpr src_dim_size = src[i];

      if (src_minor_pos >= 0) {
        CHECK(is_const_int(src[src_minor_pos], src_factor))
          << "src shape " << Array<IndexExpr>(src)
          << " does not agree with layout "
          << src_layout;
        src_dim_size *= src_factor;
      }
      dst[dst_major_pos] = src_dim_size;
      if (dst_minor_pos >= 0) {
        CHECK_GT(dst_factor, 0);
        if (const int64_t* const_src_dim_size = as_const_int(src_dim_size)) {
          CHECK_LE(dst_factor, const_src_dim_size[0])
            << "Converting " << Array<IndexExpr>(src)
            << " from " << src_layout
            << " to " << dst_layout
            << ": cannot split dimension size of "
            << src_dim_size << " by " << dst_factor;
        }
        dst[dst_major_pos] /= dst_factor;
        dst[dst_minor_pos] = dst_factor;
      }
    }
  }
  return dst;
}

std::vector<IndexExpr> ConvertLayout(
    const Array<IndexExpr>& src,
    const Layout& src_layout,
    const Layout& dst_layout) {
  std::vector<IndexExpr> ret(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    ret[i] = src[i];
  }
  return ConvertLayout(ret, src_layout, dst_layout);
}

}  // namespace relay
}  // namespace tvm
