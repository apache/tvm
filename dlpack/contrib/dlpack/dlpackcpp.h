/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpackcpp.h
 * \brief Example C++ wrapper of DLPack
 */
#ifndef DLPACK_DLPACKCPP_H_
#define DLPACK_DLPACKCPP_H_

#include <dlpack/dlpack.h>

#include <cstdint>  // for int64_t etc
#include <cstdlib>  // for free()
#include <functional>  // for std::multiplies
#include <memory>
#include <numeric>
#include <vector>

namespace dlpack {

// Example container wrapping of DLTensor.
class DLTContainer {
 public:
  DLTContainer() {
    // default to float32
    handle_.data = nullptr;
    handle_.dtype.code = kDLFloat;
    handle_.dtype.bits = 32U;
    handle_.dtype.lanes = 1U;
    handle_.ctx.device_type = kDLCPU;
    handle_.ctx.device_id = 0;
    handle_.shape = nullptr;
    handle_.strides = nullptr;
    handle_.byte_offset = 0;
  }
  ~DLTContainer() {
    if (origin_ == nullptr) {
      free(handle_.data);
    }
  }
  operator DLTensor() {
    return handle_;
  }
  operator DLTensor*() {
    return &(handle_);
  }
  void Reshape(const std::vector<int64_t>& shape) {
    shape_ = shape;
    int64_t sz = std::accumulate(std::begin(shape), std::end(shape),
                                 int64_t(1), std::multiplies<int64_t>());
    int ret = posix_memalign(&handle_.data, 256, sz);
    if (ret != 0) throw std::bad_alloc();
    handle_.shape = &shape_[0];
    handle_.ndim = static_cast<uint32_t>(shape.size());
  }

 private:
  DLTensor handle_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  // original space container, if
  std::shared_ptr<DLTContainer> origin_;
};

}  // namespace dlpack
#endif  // DLPACK_DLPACKCPP_H_
