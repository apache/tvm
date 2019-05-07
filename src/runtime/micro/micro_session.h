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
#include <memory>
#include <unordered_map>
#include "low_level_device.h"
#include "micro_common.h"
#include "device/utvm_runtime.h"
#include "target_data_layout_encoder.h"

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
  MicroSectionAllocator(dev_base_offset section_start, dev_base_offset section_end)
    : section_start_(section_start), section_end_(section_end),
      section_max_(section_start) {
  }

  /*!
   * \brief destructor
   */
  ~MicroSectionAllocator() {
  }

  /*!
   * \brief memory allocator
   * \param size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  dev_base_offset Allocate(size_t size) {
    dev_base_offset alloc_ptr = dev_base_offset(nullptr);
    if (section_max_.val_ + size < section_end_.val_) {
      alloc_ptr = section_max_;
      section_max_ = dev_base_offset(section_max_.val_ + size);
      alloc_map_[(void*)alloc_ptr.val_] = size;
    }
    return alloc_ptr;
  }

  /*!
   * \brief free prior allocation from section
   * \param type type of section to allocate in
   * \param ptr pointer to allocated memory
   * \note simple allocator scheme, more complex versions will be implemented later
   */
  void Free(dev_base_offset ptr) {
    alloc_map_.erase(reinterpret_cast<void*>(ptr.val_));
    if (alloc_map_.empty()) {
      section_max_ = section_start_;
    }
  }

  /*!
   * \brief obtain the end address of the last allocation
   * \return pointer immediately following the last allocation
   */
  dev_base_offset section_max() {
    return section_max_;
  }

 private:
  /*! \brief start address of the section */
  dev_base_offset section_start_;
  /*! \brief end address of the section */
  dev_base_offset section_end_;
  /*! \brief end address of last allocation */
  dev_base_offset section_max_;
  /*! \brief allocation map for allocation sizes */
  std::unordered_map<void*, size_t> alloc_map_;
};

class MicroSession {
 public:
  /*!
   * \brief constructor
   */
  MicroSession();

  /*!
   * \brief destructor
   */
  ~MicroSession();

  /*!
   * \brief get MicroSession global singleton
   * \return pointer to the micro session global singleton
   */
  static MicroSession* Global();

  /*!
   * \brief allocate memory in section
   * \param type type of section to allocate in
   * \param size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  dev_base_offset AllocateInSection(SectionKind type, size_t size);

  /*!
   * \brief free prior allocation from section
   * \param type type of section to allocate in
   * \param ptr pointer to allocated memory
   */
  void FreeInSection(SectionKind type, dev_base_offset ptr);

  std::string ReadString(dev_base_offset str_offset);

  /*!
   * \brief sets up init stub pointers and copies arguments for on-device execution
   * \param func address of the function to be executed
   * \param args args to the packed function
   */
  void PushToExecQueue(dev_base_offset func, TVMArgs args);

  /*!
   * \brief returns low-level device pointer
   * \note assumes low_level_device_ is initialized
   */
  // TODO(weberlo): remove &
  const std::shared_ptr<LowLevelDevice>& low_level_device() const {
    return low_level_device_;
  }

  // TODO(weberlo): add back const
  SymbolMap init_symbol_map() const {
    return init_symbol_map_;
  }

 private:
  /*! \brief low-level device pointer */
  std::shared_ptr<LowLevelDevice> low_level_device_;
  /*! \brief text section allocator */
  std::unique_ptr<MicroSectionAllocator> text_allocator_;
  /*! \brief rodata section allocator */
  std::unique_ptr<MicroSectionAllocator> rodata_allocator_;
  /*! \brief data section allocator */
  std::unique_ptr<MicroSectionAllocator> data_allocator_;
  /*! \brief bss section allocator */
  std::unique_ptr<MicroSectionAllocator> bss_allocator_;
  /*! \brief args section allocator */
  std::unique_ptr<MicroSectionAllocator> args_allocator_;
  /*! \brief stack section allocator */
  std::unique_ptr<MicroSectionAllocator> stack_allocator_;
  /*! \brief heap section allocator */
  std::unique_ptr<MicroSectionAllocator> heap_allocator_;
  /*! \brief workspace section allocator */
  std::unique_ptr<MicroSectionAllocator> workspace_allocator_;
  /*! \brief init text start address */
  dev_base_offset init_text_start_;
  /*! \brief init rodata start address */
  dev_base_offset init_rodata_start_;
  /*! \brief init data start address */
  dev_base_offset init_data_start_;
  /*! \brief init bss start address */
  dev_base_offset init_bss_start_;
  /*! \brief size of init text section */
  size_t init_text_size_;
  /*! \brief size of init rodata section */
  size_t init_rodata_size_;
  /*! \brief size of init data section */
  size_t init_data_size_;
  /*! \brief size of init bss section */
  size_t init_bss_size_;
  /*! \brief symbol map for init stub */
  SymbolMap init_symbol_map_;
  /*! \brief path to init stub source code */
  std::string init_binary_path_;
  /*! \brief offset of the init stub entry function */
  dev_base_offset utvm_main_symbol_addr_;
  /*! \brief offset of the init stub exit breakpoint */
  dev_base_offset utvm_done_symbol_addr_;

  /*!
   * \brief sets up and loads init stub into the low-level device memory
   */
  void LoadInitStub();

  /*!
   * \brief sets the init stub binary path
   * \param path to init stub binary
   */
  void SetInitBinaryPath(std::string path);

  /*!
   * \brief writes arguments to the host-side buffer of `encoder`
   * \param encoder encoder being used to write `args`
   * \param args pointer to the args to be written
   * \return device address of the allocated args
   */
  dev_addr EncoderWrite(TargetDataLayoutEncoder* encoder, UTVMArgs* args);

  /*!
   * \brief writes a `TVMArray` to the host-side buffer of `encoder`
   * \param encoder encoder being used to write `arr`
   * \param arr pointer to the TVMArray to be written
   * \return device address of the allocated `TVMArray`
   */
  dev_addr EncoderWrite(TargetDataLayoutEncoder* encoder, TVMArray* arr);
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_SESSION_H_
