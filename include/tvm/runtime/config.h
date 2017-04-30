/*!
 *  Copyright (c) 2017 by Contributors
 * \file config.h
 * \brief Runtime library related configurations.
 */
#ifndef TVM_RUNTIME_CONFIG_H_
#define TVM_RUNTIME_CONFIG_H_

/*!
 *\brief whether to use CUDA runtime
 */
#ifndef TVM_CUDA_RUNTIME
#define TVM_CUDA_RUNTIME 1
#endif

/*!
 *\brief whether to use opencl runtime
 */
#ifndef TVM_OPENCL_RUNTIME
#define TVM_OPENCL_RUNTIME 0
#endif

/*!
 *\brief whether to use metal runtime
 */
#ifndef TVM_METAL_RUNTIME
#define TVM_METAL_RUNTIME 0
#endif

#endif  // TVM_RUNTIME_CONFIG_H_
