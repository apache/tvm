/*!
*  Copyright (c) 2017 by Contributors
* \file compilation.h
* \brief Functions for compiling ops.
*/
#ifndef TVM_COMPILATION_H_
#define TVM_COMPILATION_H_

#include <string>
#include "./tvm/c_dsl_api.h"
#include "./tvm/runtime/packed_func.h"
#include "./tvm/schedule_pass.h"
#include "./tvm/lowered_func.h"

namespace tvm {

namespace compilation {

/*!
* \brief Container for target device information.
* Use target_llvm, target_cuda etc functions instead of constructing directly.
*/
struct Target {
    /*! \brief The name of the target device */
    std::string targetName;
    /*! \brief The type of the target device */
    DLDeviceType deviceType;
    /*! \brief The maximum threads that a schedule should use for this device */
    int max_num_threads;
    /*! \brief The warp size that should be used by the LowerThreadAllreduce pass */
    int thread_warp_size;
    /*! \brief Keys for this target */
    std::unordered_set<std::string> keys;
    /*! \brief Options for this target */
    std::unordered_set<std::string> options;


    Target(std::string targetName, DLDeviceType deviceType, int max_num_threads,
        int thread_warp_size, const std::unordered_set<std::string>& keys,
        const std::unordered_set<std::string>& options) {
        this->targetName = targetName;
        this->deviceType = deviceType;
        this->max_num_threads = max_num_threads;
        this->thread_warp_size = thread_warp_size;
        this->keys = keys;
        this->options = options;
    }

    /*! \return the full device string to pass to codegen::Build */
    std::string str() const {
        std::ostringstream result;
        result << targetName;
        for (const auto &x : options) {
            result << " " << x;
        }
        return result.str();
    }

    /*! \return True iff this target has the given key */
    bool hasKey(std::string key) const {
        return keys.count(key) > 0;
    }
};

/*! \return A target for LLVM */
Target target_llvm() {
    std::unordered_set<std::string> keys({ "llvm", "cpu" });
    std::unordered_set<std::string> options;
    return Target("llvm", kDLCPU, 512, 1, keys, options);
}

/*! \return A target for CUDA */
Target target_cuda() {
    std::unordered_set<std::string> keys({ "cuda", "gpu" });
    std::unordered_set<std::string> options;
    return Target("cuda", kDLGPU, 512, 32, keys, options);
}

/*! \return A target for ROCm */
Target target_rocm() {
    std::unordered_set<std::string> keys({ "rocm", "gpu" });
    std::unordered_set<std::string> options;
    return Target("rocm", kDLROCM, 256, 1, keys, options);
}

/*! \return A target for Metal */
Target target_metal() {
    std::unordered_set<std::string> keys({ "gpu" });
    std::unordered_set<std::string> options;
    return Target("metal", kDLMetal, 256, 1, keys, options);
}

/*! \return A target for rasp */
Target target_rasp() {
    std::unordered_set<std::string> keys({ "llvm", "cpu" });
    std::unordered_set<std::string> options({
        "-device=rasp",
        "-mtriple=armv7l-none-linux-gnueabihf",
        "-mcpu=cortex-a53",
        "-mattr=+neon"
    });
    return Target("llvm", kDLCPU, 512, 1, keys, options);
}

/*! \return A target for stackvm */
Target target_stackvm() {
    std::unordered_set<std::string> keys({ "stackvm", "cpu" });
    std::unordered_set<std::string> options;
    return Target("stackvm", kDLCPU, 512, 1, keys, options);
}


bool LLVMEnabled() {
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.llvm_target_enabled");
    if (pf == nullptr) return false;
    return (*pf)("llvm");
}

/*! \return The default host target for a given device target */
Target default_target_host(Target target) {
    if (target.deviceType == kDLCPU) {
        return target;
    } else {
        if (LLVMEnabled()) {
            return target_llvm();
        } else {
            return target_stackvm();
        }
    }
}


/*!
* \brief Container for build configuration options
*/
struct BuildConfig {
    int data_alignment = -1;
    int offset_factor = 0;

    int double_buffer_split_loop = 1;
    int auto_unroll_max_step = 0;
    int auto_unroll_max_depth = 8;
    int auto_unroll_max_extent = 0;
    bool unroll_explicit = true;

    bool restricted_func = true;

    bool detect_global_barrier = false;

    BuildConfig() {
    }
};


/*! \brief Convenience function for getting attributes */
TVMValue GetAttr(NodeRef node, std::string attrName) {
    TVMValue attrValue;
    int typeCode, success;

    TVMNodeGetAttr((NodeHandle)&node, attrName.c_str(), &attrValue, &typeCode, &success);
    return attrValue;
}

/*! \brief Convenience function for getting handle attributes */
template<typename T>
T GetAttrHandle(NodeRef node, std::string attrName) {
    auto attrValue = GetAttr(node, attrName);
    return *(reinterpret_cast<T*>(attrValue.v_handle));
}


/*!
* \brief Build a Stmt given a schedule, args and binds. This function runs the IR passes.
* \param sch The schedule to build.
* \param args The arguments for the schedule.
* \param binds Buffer assignments.
* \param loopPartition True if the LoopPartition pass should be included.
* \param argListOut Returns the arguments for the Stmt.
* \param config The build configuration.
* \return The built Stmt.
*/
EXPORT Stmt BuildStmt(Schedule sch, Array<Tensor> args, std::unordered_map<Tensor, Buffer> binds,
    bool loopPartition, Array<NodeRef> *argListOut, const BuildConfig& config);

/*!
* \brief Build a LoweredFunc given a schedule, args and binds
* \param sch The schedule to lower.
* \param args The arguments to the function.
* \param name The name of the lowered function.
* \param binds Buffer assignments.
* \param config The build configuration.
* \return The lowered function.
*/
EXPORT LoweredFunc Lower(Schedule sch, Array<Tensor> args, std::string name,
    std::unordered_map<Tensor, Buffer> binds, const BuildConfig& config);

/*!
* \brief Build a device and host module for a specific target from an array of lowered functions.
* \param funcs The functions to be built.
* \param target The target device to build for.
* \param targetHost The target for building host code.
* \return The built module.
*/
EXPORT runtime::Module BuildModule(Array<LoweredFunc> funcs, const Target& target,
    const Target& targetHost, const BuildConfig& config);

}  // namespace compilation
}  // namespace tvm

#endif  // TVM_COMPILATION_H_
