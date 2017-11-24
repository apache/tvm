/*!
*  Copyright (c) 2017 by Contributors
* \file build_module.h
* \brief Functions for compiling ops.
*/
#ifndef TVM_BUILD_MODULE_H_
#define TVM_BUILD_MODULE_H_

#include <string>
#include "./tvm/c_dsl_api.h"
#include "./tvm/runtime/packed_func.h"
#include "./tvm/schedule_pass.h"
#include "./tvm/lowered_func.h"

namespace tvm {

/*!
* \brief Container for target device information.
* Use target_llvm, target_cuda etc functions instead of constructing directly.
*/
struct Target {
    /*! \brief The name of the target device */
    std::string target_name;
    /*! \brief The type of the target device */
    DLDeviceType device_type;
    /*! \brief The maximum threads that a schedule should use for this device */
    int max_num_threads = 1;
    /*! \brief The warp size that should be used by the LowerThreadAllreduce pass */
    int thread_warp_size = 1;
    /*! \brief Keys for this target */
    std::unordered_set<std::string> keys;
    /*! \brief Options for this target */
    std::unordered_set<std::string> options;


    Target(const std::string& target_name, DLDeviceType device_type, int max_num_threads,
        int thread_warp_size, const std::unordered_set<std::string>& keys,
        const std::unordered_set<std::string>& options) {
        this->target_name = target_name;
        this->device_type = device_type;
        this->max_num_threads = max_num_threads;
        this->thread_warp_size = thread_warp_size;
        this->keys = keys;
        this->options = options;
    }

    /*! \return the full device string to pass to codegen::Build */
    std::string str() const;
};

namespace target {
/*! \return A target for LLVM */
EXPORT Target llvm();

/*! \return A target for CUDA */
EXPORT Target cuda();

/*! \return A target for ROCm */
EXPORT Target rocm();

/*! \return A target for Metal */
EXPORT Target metal();

/*! \return A target for rasp */
EXPORT Target rasp();

/*! \return A target for stackvm */
EXPORT Target stackvm();

}  // namespace target


/*!
* \brief Container for build configuration options
*/
struct BuildConfig {
    /*!
     * \brief The data alignment to use when constructing buffers. If this is set to
     * -1, then TVM's internal default will be used
     */
    int data_alignment = -1;
    /*!
     * \brief The offset factor to use when constructing buffers. If this is set to
     * 0, then the offset field is not used.
     */
    int offset_factor = 0;

    /*!
     * \brief Splitting factor for loop splitting. If this is set to zero, no splitting will be
     * done. Otherwise, a split will be done with this factor and the inner loop will be unrolled.
     */
    int double_buffer_split_loop = 1;
    /*! \brief Threshold of number of steps in the loop to be automatically unrolled */
    int auto_unroll_max_step = 0;
    /*! \brief The maximum nested level of loops that can be automatically unrolled */
    int auto_unroll_max_depth = 8;
    /*! \brief The maximum extent of loop that will be unrolled */
    int auto_unroll_max_extent = 0;
    /*!
     * \brief Whether to explicitly unroll the loop. If set to false, the unroll hint will
     * be passed to the CodeGen phase. Set to true if CodeGen supports unroll pragma.
     */
    bool unroll_explicit = true;

    /*! \brief Set to true if buffer arguments do not overlap. This enables more optimization. */
    bool restricted_func = true;

    /*! \brief Whether to detect global barrier */
    bool detect_global_barrier = false;

    BuildConfig() {
    }
};



/*!
* \brief Build a LoweredFunc given a schedule, args and binds
* \param sch The schedule to lower.
* \param args The arguments to the function.
* \param name The name of the lowered function.
* \param binds Buffer assignments.
* \param config The build configuration.
* \return The lowered function.
*/
EXPORT LoweredFunc lower(Schedule sch, const Array<Tensor>& args, const std::string& name,
    const std::unordered_map<Tensor, Buffer>& binds, const BuildConfig& config);

/*!
* \brief Build a device and host module for a specific target from an array of lowered functions.
* \param funcs The functions to be built.
* \param target The target device to build for.
* \param targetHost The target for building host code. If null, a suitable default will be used.
* \param config The build configuration.
* \return The built module.
*/
EXPORT runtime::Module build(const Array<LoweredFunc>& funcs, const Target& target,
    Target* target_host, const BuildConfig& config);

}  // namespace tvm

#endif  // TVM_BUILD_MODULE_H_
