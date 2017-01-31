/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_module.h
 * \brief Execution handling of OPENCL kernels
 */
#ifndef TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
#define TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Handle execution of OPENCL kernels as PackedFunc.
 *  It wraps around driver API to work with OPENCL runtime API.
 */
class OpenCLModule {
 public:
  /*!
   * \brief Get OpenCL Kernel launch wrapped as PackedFunc
   * \param func_name The name of the function.
   * \param arg_types The type of each argument in the function.
   * \param thread_axis_tags The tag sequence of the thread axis.
   */
  PackedFunc GetPackedFunc(
      const std::string& func_name,
      const std::vector<TVMType> arg_types,
      const std::vector<std::string> thread_axis_tags) const;
  /*!
   * \brief create a OpenCL module from data.
   * \param source The module data.
   */
  static OpenCLModule CreateWithSource(std::string source);
  /*! \brief hidden internal data structure. */
  class Internal;

 private:
  std::shared_ptr<Internal> ptr_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
