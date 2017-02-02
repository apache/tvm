/*!
 *  Copyright (c) 2016 by Contributors
 * \file codegen.h
 * \brief Collection of Lowlevel IR pass to codegen.
 */
#ifndef TVM_CODEGEN_H_
#define TVM_CODEGEN_H_

#include <string>
#include "./base.h"
#include "./expr.h"
#include "./lowered_func.h"
#include "./runtime/packed_func.h"


namespace tvm {
/*! \brief namespace for lowlevel IR pass and codegen */
namespace codegen {
// use packed function from runtime.
using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

/*!
 * \brief Build a stack VM function.
 * \param func The LoweredFunc to be build
 * \param device_funcs The additional device functions
 * \return A packed function representing the func.
 */
PackedFunc BuildStackVM(
    LoweredFunc func,
    const std::unordered_map<LoweredFunc, PackedFunc>& device_funcs);

/*!
 * \brief Build a CUDA function with NVRTC
 *
 * \param fsplits The LoweredFuncs to be build (after SplitHostDevice)
 *  The first element is the host function, followed by device functions.
 * \param host_mode The host side compilation mode:
 *   - "stackvm": use stack vm to interpret host side code.
 */
PackedFunc BuildNVRTC(Array<LoweredFunc> fsplits, std::string host_mode);

/*!
 * \brief Build a OpenCL function.
 *
 * \param fsplits The LoweredFuncs to be build (after SplitHostDevice)
 *  The first element is the host function, followed by device functions.
 * \param host_mode The host side compilation mode:
 *   - "stackvm": use stack vm to interpret host side code.
 */
PackedFunc BuildOpenCL(Array<LoweredFunc> fsplits, std::string host_mode);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_H_
