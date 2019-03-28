/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_session.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_SESSION_H_
#define TVM_RUNTIME_MICRO_MICRO_SESSION_H_

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <mutex>
#include "low_level_device.h"
#include "allocator_stream.h"
#include "micro_common.h"
#include "device/utvm_runtime.h"

namespace tvm {
namespace runtime {
/*!
 * \brief allocator for a on-device memory section
 */
class MicroSectionAllocator {
 public:
  /*!
   * \brief constructor that specifies section boundaries
   * \param section_start start address of the section
   * \param section_end end address of the section (non inclusive)
   */
  MicroSectionAllocator(void* section_start, void* section_end);

  /*!
   * \brief memory allocator
   * \param size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  void* Allocate(size_t size);

  /*!
   * \brief free prior allocation from section
   * \param type type of section to allocate in
   * \param ptr pointer to allocated memory
   */
  void Free(void* ptr);

 private:
  /*! \brief start address of the section */
  void* section_start_;
  /*! \brief end address of the section */
  void* section_end_;
  /*! \brief end address of last allocation */
  void* section_max_;
  /*! \brief allocation map for allocation sizes */
  std::unordered_map<void*, size_t> alloc_map_;
};

class MicroSession {
 public:
  /*!
   * \brief destructor
   */
  ~MicroSession();

  /*!
   * \brief get MicroSession global singleton
   * \return pointer to the micro session global singleton
   */
  static const MicroSession* Global();

  /*!
   * \brief allocate memory in section
   * \param type type of section to allocate in
   * \param size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  void* AllocateInSection(SectionKind type, size_t size);

  /*!
   * \brief free prior allocation from section
   * \param type type of section to allocate in
   * \param ptr pointer to allocated memory
   */
  void FreeInSection(SectionKind type, void* ptr);

  /*!
   * \brief sets up init stub pointers and copies arguments for on-device execution
   * \param func address of the function to be executed
   * \param args args to the packed function
   */
  void PushToExecQueue(void* func, TVMArgs args);

  /*!
   * \brief returns low-level device pointer
   * \note assumes low_level_device_ is initialized
   */
  const std::shared_ptr<LowLevelDevice> low_level_device() const {
    return low_level_device_;
  }

 private:
  /*! \brief low-level device pointer */
  std::shared_ptr<LowLevelDevice> low_level_device_;
  /*! \brief text section allocator */
  MicroSectionAllocator text_allocator_;
  /*! \brief data section allocator */
  MicroSectionAllocator data_allocator_;
  /*! \brief bss section allocator */
  MicroSectionAllocator bss_allocator_;
  /*! \brief args section allocator */
  MicroSectionAllocator args_allocator_;
  /*! \brief stack section allocator */
  MicroSectionAllocator stack_allocator_;
  /*! \brief heap section allocator */
  MicroSectionAllocator heap_allocator_;
  /*! \brief workspace section allocator */
  MicroSectionAllocator workspace_allocator_;
  /*! \brief symbol map for init stub */
  std::unordered_map<std::string, void*> init_symbol_map_;

  /*!
   * \brief sets up and loads init stub into the low-level device memory
   */
  void SetupInitStub();

  /*!
   * \brief writes arguments to args section using allocator_stream
   */
  void AllocateTVMArgs(TVMArgs args);
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_SESSION_H_
