/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_module.h
 * \brief Execution handling of CUDA kernels
 */
#ifndef TVM_RUNTIME_CUDA_CUDA_MODULE_H_
#define TVM_RUNTIME_CUDA_CUDA_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Handle execution of CUDA kernels as PackedFunc.
 *  It wraps around driver API to work with CUDA runtime API.
 */
class CUDAModule {
 public:
  /*!
   * \brief Get CUDA Kernel launch wrapped as PackedFunc
   * \param func_name The name of the function.
   * \param arg_types The type of each argument in the function.
   * \param thread_axis_tags The tag sequence of the thread axis.
   */
  PackedFunc GetPackedFunc(
      const std::string& func_name,
      const std::vector<TVMType> arg_types,
      const std::vector<std::string> thread_axis_tags) const;
  /*!
   * \brief create a cuda module from data.
   * \param data The module data.
   */
  static CUDAModule Create(std::string data);
  /*! \brief hidden internal data structure. */
  class Internal;
  /*! \brief Maximum number of GPU supported in CUDAModule */
  static constexpr const int kMaxNumGPUs = 32;

 private:
  std::shared_ptr<Internal> ptr_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CUDA_CUDA_MODULE_H_
