/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn_common.h
 * \brief Common utilities for nn ops.
 */
#ifndef NNVM_TOP_NN_NN_COMMON_H_
#define NNVM_TOP_NN_NN_COMMON_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <nnvm/top/nn.h>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

namespace nnvm {
namespace top {

template<typename ParamType>
inline uint32_t UseBiasNumInputs(const NodeAttrs& attrs) {
  const ParamType& param = get<ParamType>(attrs.parsed);
  return param.use_bias ? 3 : 2;
}

template<typename ParamType>
inline std::vector<std::string> UseBiasListInputNames(const NodeAttrs& attrs) {
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  if (param.use_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

/*!
 * \brief Convert shape in src_layout to shape in dst_layout
 * \param src original shape
 * \param src_layout layout of original shape
 * \param dst_layout target layout
 * \return shape in target layout
 */
inline TShape ConvertLayout(TShape src, int src_layout, int dst_layout, bool is_weight = false) {
  if (src_layout == dst_layout) return src;
  TShape dst = src;
  if (src.ndim() == 3) {
    switch (src_layout) {
      case kNCW: break;
      case kNWC: {
        std::swap(dst[1], dst[2]);
        break;
      }
      default: {
        LOG(FATAL) << "inavlid layout for 3d shape" << src_layout;
      }
    }
    switch (dst_layout) {
      case kNCW: break;
      case kNWC: {
        std::swap(dst[1], dst[2]);
        break;
      }
      default: {
        LOG(FATAL) << "inavlid layout for 3d shape" << dst_layout;
      }
    }
  } else if (src.ndim() == 4) {
    switch (src_layout) {
      case kNCHW: break;
      case kNHWC: {
        if (is_weight) {
           dst[2] = src[0];
           dst[3] = src[1];
           dst[1] = src[2];
           dst[0] = src[3];
        } else {
           dst[2] = src[1];
           dst[3] = src[2];
           dst[1] = src[3];
        }
        break;
      }
      default: {
        LOG(FATAL) << "inavlid layout for 4d shape" << src_layout;
      }
    }
    src = dst;
    switch (dst_layout) {
      case kNCHW: break;
      case kNHWC: {
        if (is_weight) {
            dst[0] = src[2];
            dst[1] = src[3];
            dst[2] = src[1];
            dst[3] = src[0];
        } else {
            dst[1] = src[2];
            dst[2] = src[3];
            dst[3] = src[1];
        }
        break;
      }
      default: {
        LOG(FATAL) << "inavlid layout for 4d shape" << dst_layout;
      }
    }
  } else if (src.ndim() == 5) {
    switch (src_layout) {
      case kNCDHW: break;
      case kNDHWC: {
        dst[2] = src[1];
        dst[3] = src[2];
        dst[4] = src[3];
        dst[1] = src[4];
        break;
      }
      default: {
        LOG(FATAL) << "inavlid layout for 5d shape" << src_layout;
      }
    }
    src = dst;
    switch (dst_layout) {
      case kNCDHW: break;
      case kNDHWC: {
        dst[1] = src[2];
        dst[2] = src[3];
        dst[3] = src[4];
        dst[4] = src[1];
        break;
      }
      default: {
        LOG(FATAL) << "inavlid layout for 5d shape" << dst_layout;
      }
    }
  } else {
    LOG(FATAL) << "no layout option for " << dst.ndim() << " dimensions";
  }
  return dst;
}

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_NN_NN_COMMON_H_
