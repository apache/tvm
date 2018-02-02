/*!
 *  Copyright (c) 2017 by Contributors
 * \file metal_module.h
 * \brief Execution handling of Metal kernels
 */
#ifndef TVM_RUNTIME_VULKAN_VULKAN_MODULE_H_
#define TVM_RUNTIME_VULKAN_VULKAN_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <dmlc/type_traits.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "../meta_data.h"

namespace tvm {
namespace runtime {
/*! \brief Maximum number of GPU supported in VulkanModule. */
static constexpr const int kVulkanMaxNumDevice = 8;

/*! \brief TVM Vulkan binary pack magic number */
static constexpr const int kVulkanModuleMagic = 0x02700027;

/*!
 * \brief A single VK shader program
 *
 *  Due to the global resource declaration.
 *  Current SPIRV only allows one entry program per shader,
 *  making it less useful for a Module like system.
 *
 *  Instead we pass in map of str->VulkanShader until
 *  there is a native solution available.
 */
struct VulkanShader {
  /*! \brief header flag */
  uint32_t flag{0};
  /*! \brief Data segment */
  std::vector<uint32_t> data;

  void Save(dmlc::Stream *writer) const;
  bool Load(dmlc::Stream *reader);
};

/*!
 * \brief create a metal module from data.
 *
 * \param pmap The program map.
 * \param fmap The function information map.
 * \param source Optional, source code.
 */
Module VulkanModuleCreate(
    std::unordered_map<std::string, VulkanShader> smap,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source);
}  // namespace runtime
}  // namespace tvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::VulkanShader, true);
}  // namespace dmlc

#endif  // TVM_RUNTIME_VULKAN_VULKAN_MODULE_H_
