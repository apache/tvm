/*!
 *  Copyright (c) 2018 by Contributors
 * \file runtime.cc
 * \brief Generic VTA runtime in C++11.
 *
 *  The runtime depends on specific instruction
 *  stream spec as specified in hw_spec.h
 */
#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <vta/runtime.h>
#include <dmlc/logging.h>

#include <cassert>
#include <cstring>
#include <vector>
#include <thread>
#include <memory>
#include <atomic>

namespace vta {

// Avoid bad configurations.
static_assert(VTA_UOP_WIDTH == sizeof(VTAUop) * 8,
              "VTA_UOP_WIDTH do not match VTAUop size");

/*! \brief Enable coherent access between VTA and CPU. */
static const bool kBufferCoherent = true;

/*!
 * \brief Data buffer represents data on CMA.
 */
struct DataBuffer {
  /*! \return Virtual address of the data. */
  void* virt_addr() const {
    return data_;
  }
  /*! \return Physical address of the data. */
  uint32_t phy_addr() const {
    return phy_addr_;
  }
  /*!
   * \brief Invalidate the cache of given location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void InvalidateCache(size_t offset, size_t size) {
    if (!kBufferCoherent) {
      VTAInvalidateCache(phy_addr_ + offset, size);
    }
  }
  /*!
   * \brief Invalidate the cache of certain location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void FlushCache(size_t offset, size_t size) {
    if (!kBufferCoherent) {
      VTAFlushCache(phy_addr_ + offset, size);
    }
  }
  /*!
   * \brief Allocate a buffer of a given size.
   * \param size The size of the buffer.
   */
  static DataBuffer* Alloc(size_t size) {
    void* data = VTAMemAlloc(size, 1);
    CHECK(data != nullptr);
    DataBuffer* buffer = new DataBuffer();
    buffer->data_ = data;
    buffer->phy_addr_ = VTAMemGetPhyAddr(data);
    return buffer;
  }
  /*!
   * \brief Free the data buffer.
   * \param buffer The buffer to be freed.
   */
  static void Free(DataBuffer* buffer) {
    VTAMemFree(buffer->data_);
    delete buffer;
  }
  /*!
   * \brief Create data buffer header from buffer ptr.
   * \param buffer The buffer pointer.
   * \return The corresponding data buffer header.
   */
  static DataBuffer* FromHandle(const void* buffer) {
    return const_cast<DataBuffer*>(
        reinterpret_cast<const DataBuffer*>(buffer));
  }

 private:
  /*! \brief The internal data. */
  void* data_;
  /*! \brief The physical address of the buffer, excluding header. */
  uint32_t phy_addr_;
};

/*!
 * \brief Micro op kernel.
 *  Contains functions to construct the kernel with prefix Push.
 */
class UopKernel {
 public:
  /*! \brief Loop information. */
  struct LoopEntry {
    uint32_t extent;
    uint32_t dst_factor;
    uint32_t src_factor;
    uint32_t wgt_factor;
  };
  /*!
   * \brief Construct UopKernel with signature.
   * \param signature The pointer to signature.
   * \param nbytes Number of bytes.
   */
  UopKernel(const char* signature, int nbytes)
      : signature_(signature, signature + nbytes) {
  }
  /*!
   * \brief Verify if the signature is correct.
   * \param signature Signature ptr.
   * \param nbytes Number of bytes.
   */
  bool MatchSignature(void* signature, int nbytes) const {
    if (static_cast<size_t>(nbytes) != signature_.size()) return false;
    return memcmp(signature, signature_.data(), nbytes) == 0;
  }
  /*! \return Whether the kernel is cached in SRAM. */
  bool cached() const {
    return sram_begin_ != sram_end_;
  }
  /*! \return The length of the micro op sequence. */
  size_t size() const {
    return seq_.size();
  }
  /*! \return The micro-op data. */
  const VTAUop* data() const {
    return seq_.data();
  }
  /*! \return The loop structure. */
  const std::vector<LoopEntry>& loop() const {
    return loop_;
  }
  /*!
   * \brief Declare loop start.
   * \param extent The loop extent.
   * \param dst_factor Loop factor of accum index.
   * \param src_factor Loop factor of input index
   * \param wgt_factor Loop factor of weight index.
   */
  void PushLoopBegin(uint32_t extent,
                     uint32_t dst_factor,
                     uint32_t src_factor,
                     uint32_t wgt_factor) {
    LoopEntry le;
    le.extent = extent;
    le.dst_factor = dst_factor;
    le.src_factor = src_factor;
    le.wgt_factor = wgt_factor;
    CHECK_EQ(seq_.size(), 0U);
    CHECK_LT(loop_.size(), 2U);
    loop_.push_back(le);
    ++loop_ptr_;
  }
  /*!
   * \brief Declare loop end.
   */
  void PushLoopEnd() {
    --loop_ptr_;
  }
  /*!
   * \brief Push micro op into kernel.
   * \param mode Set to GEMM mode if set to 0, ALU mode is set to 1.
   * \param reset_out Resets the accum to 0.
   * \param dst_index The accum memory index.
   * \param src_index The input memory (gemm) / accum memory (alu) index.
   * \param wgt_index The weight memory index.
   * \param opcode The ALU opcode.
   * \param use_imm Use immediate in ALU mode if set to true.
   * \param imm_val Immediate value in ALU mode.
   */
  void Push(uint32_t mode,
            uint32_t reset_out,
            uint32_t dst_index,
            uint32_t src_index,
            uint32_t wgt_index,
            uint32_t opcode,
            uint32_t use_imm,
            int32_t imm_val) {
    // The loop nest structure
    VerifyDep(dst_index);
    VTAUop op;
    op.dst_idx = dst_index;
    op.src_idx = src_index;
    op.wgt_idx = wgt_index;
    seq_.push_back(op);
    // Ensure that mode is consistent if set
    if (mode_ == 0xFFFFFFFF) {
      mode_ = mode;
    } else {
      CHECK(mode_ == mode);
    }
    // Set reset_out field if unset
    if (reset_out_ == 0xFFFFFFFF) {
      reset_out_ = reset_out;
    } else {
      CHECK(reset_out_ == reset_out);
    }
    // Check kernel op and imm/imm_val in ALU mode
    if (mode == 1) {
      if (opcode_ == 0xFFFFFFFF) {
        opcode_ = opcode;
        use_imm_ = use_imm;
        imm_val_ = imm_val;
      } else {
        CHECK(opcode_ == opcode);
        CHECK(use_imm_ == use_imm);
        CHECK(imm_val_ == imm_val);
      }
    }
  }
  /*! \brief Dump kernel micro ops to stdout. */
  void Dump() {
    uint32_t size = seq_.size();
    printf("There are %u uops\n", size);
    for (uint32_t i = 0; i < size; ++i) {
      printf("[%04u]\t acc=%u, inp=%u, wgt=%u\n",
             i,
             seq_[i].dst_idx,
             seq_[i].src_idx,
             seq_[i].wgt_idx);
    }
    printf("\n");
  }

 public:
  // The kernel's mode, opcode, immediate setting and value
  uint32_t mode_{0xFFFFFFFF};  // UOP type: 0xFFFFFFFF - unset, 0 - GEMM, 1 - ALU
  uint32_t opcode_{0xFFFFFFFF};
  uint32_t reset_out_{0xFFFFFFFF};
  bool use_imm_{false};
  int16_t imm_val_{0};

 private:
  // Verify that we don't write to the same acc_mem index two cycles in a row
  void VerifyDep(uint32_t dst_index) {
    size_t step = std::min(static_cast<size_t>(2U), seq_.size());
    for (size_t i = seq_.size() - step; i < seq_.size(); ++i) {
      CHECK(seq_[i].dst_idx != dst_index);
    }
  }
  // The uop buffer
  template<int, bool, bool>
  friend class UopQueue;
  friend class CommandQueue;
  // SRAM location if begin != end.
  uint32_t sram_begin_{0};
  uint32_t sram_end_{0};
  // The signature used for verification
  std::vector<char> signature_;
  // Internal sequence
  std::vector<VTAUop> seq_;
  // The loop nest structure specific to ALU instructions
  std::vector<LoopEntry> loop_;
  // The loop pointer
  size_t loop_ptr_{0};
};

/*!
 * \brief Base class of all queues to send and recv serial data.
 */
class BaseQueue {
 public:
  ~BaseQueue() {
    if (dram_buffer_ != nullptr) {
      VTAMemFree(dram_buffer_);
    }
  }
  /*! \return Content of DRAM buffer. */
  char* dram_buffer() const {
    return dram_buffer_;
  }
  /*! \return Physical address of DRAM. */
  uint32_t dram_phy_addr() const {
    return dram_phy_addr_;
  }
  /*! \return Whether there is pending information. */
  bool pending() const {
    return sram_begin_ != sram_end_;
  }
  /*! \brief Initialize the space of the buffer. */
  void InitSpace(uint32_t elem_bytes, uint32_t max_bytes, bool coherent, bool always_cache) {
    coherent_ = coherent;
    always_cache_ = always_cache;
    elem_bytes_ = elem_bytes;
    dram_buffer_ = static_cast<char*>(VTAMemAlloc(
        max_bytes, coherent || always_cache_));
    CHECK(dram_buffer_ != nullptr);
    dram_phy_addr_ = VTAMemGetPhyAddr(dram_buffer_);
  }
  /*!
   * \brief Reset the pointer of the buffer.
   *  Set SRAM pointer to be the current end.
   */
  void Reset() {
    dram_begin_ = dram_end_ = 0;
    sram_begin_ = sram_end_;
  }
  void AutoReadBarrier() {
    ReadBarrier(elem_bytes_ * 8, 0, dram_end_);
  }
  /*! \brief Writer barrier to make sure that data written by CPU is visible to VTA. */
  void ReadBarrier(uint32_t elem_bits, uint32_t dram_begin, uint32_t dram_extent) {
    if (!coherent_ && always_cache_ && dram_extent != 0) {
      dram_begin = dram_begin * elem_bits / 8;
      dram_extent = dram_extent * elem_bits / 8;
      VTAFlushCache(dram_phy_addr_ + dram_begin,
                    dram_extent);
    }
  }
  /*! \brief Read barrier to make sure that data written by VTA is visible to CPU. */
  void WriteBarrier(uint32_t elem_bits, uint32_t dram_begin, uint32_t dram_extent) {
    if (!coherent_ && always_cache_ && dram_extent != 0) {
      dram_begin = dram_begin * elem_bits / 8;
      dram_extent = dram_extent * elem_bits / 8;
      VTAInvalidateCache(dram_phy_addr_ + dram_begin,
                         dram_extent);
    }
  }

 protected:
  // Cache coherence access
  bool coherent_{false};
  // Make the buffer cacheable
  bool always_cache_{false};
  // Element bytes
  uint32_t elem_bytes_{0};
  // Begin location of current SRAM read in FIFO mode
  uint32_t sram_begin_{0};
  // End location of current SRAM write in FIFO mode
  uint32_t sram_end_{0};
  // The current pending offset in DRAM in FIFO mode
  uint32_t dram_begin_{0};
  // The current pending offset in DRAM in FIFO mode
  uint32_t dram_end_{0};
  // The buffer in DRAM
  char* dram_buffer_{nullptr};
  // Physics address of the buffer
  uint32_t dram_phy_addr_;
};

/*!
 * \brief Micro op buffer that manages the micro op cache.
 */
template<int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class UopQueue : public BaseQueue {
 public:
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
  }
  // Push data to the queue
  template<typename FAutoSync>
  void Push(UopKernel* kernel, FAutoSync fautosync) {
    if (kernel->cached()) return;
    size_t num_op = kernel->size();
    if (dram_end_ + num_op > kMaxElems) {
      fautosync();
      CHECK(dram_end_ <= kMaxElems);
    }
    CHECK(num_op <= kMaxNumUop);
    uint32_t uop_begin = 0;
    if (sram_end_ + num_op > kMaxNumUop) {
      // Need to evict
      cache_ptr_ = 0;
      sram_begin_ = 0;
      sram_end_ = num_op;
    } else {
      uop_begin = sram_end_;
      sram_end_ += num_op;
    }
    // Simple eviction policy
    uint32_t evict_begin = cache_ptr_;
    for (; cache_ptr_ < cache_.size(); ++cache_ptr_) {
      if (cache_[cache_ptr_]->sram_begin_ >= sram_end_) break;
      cache_[cache_ptr_]->sram_begin_ = 0;
      cache_[cache_ptr_]->sram_end_ = 0;
    }
    memcpy(dram_buffer_ + dram_end_ * kElemBytes,
           kernel->data(),
           num_op * kElemBytes);
    dram_end_ += num_op;
    kernel->sram_begin_ = uop_begin;
    kernel->sram_end_ = sram_end_;
    CHECK(kernel->cached());
    CHECK(uop_begin != sram_end_);
    cache_.insert(cache_.begin() + cache_ptr_, kernel);
    cache_.erase(cache_.begin() + evict_begin, cache_.begin() + cache_ptr_);
    cache_ptr_ = evict_begin + 1;
  }
  // Flush as weight load
  void FlushUopLoad(VTAMemInsn* insn) {
    if (sram_begin_ != sram_end_) {
      CHECK((dram_end_ - dram_begin_) == (sram_end_ - sram_begin_));
      insn->memory_type = VTA_MEM_ID_UOP;
      insn->sram_base = sram_begin_;
      insn->dram_base = dram_phy_addr_ / kElemBytes + dram_begin_;
      insn->y_size = 1;
      insn->x_size = (dram_end_ - dram_begin_);
      insn->x_stride = (dram_end_ - dram_begin_);
      insn->y_pad_0 = 0;
      insn->y_pad_1 = 0;
      insn->x_pad_0 = 0;
      insn->x_pad_1 = 0;
      // Reset indices
      sram_begin_ = sram_end_;
      dram_begin_ = dram_end_;
    }
  }

 private:
  // Cache pointer
  uint32_t cache_ptr_{0};
  // Cached ring, sorted by sram_begin
  std::vector<UopKernel*> cache_;
  // Constants
  static constexpr int kElemBytes = sizeof(VTAUop);
  static constexpr int kMaxNumUop = VTA_UOP_BUFF_DEPTH;
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};

// Internal kernel structure
class UopKernelMap {
 public:
  // Simple hash map
  UopKernel** Get(void* signature,
                  int nbytes) {
    uint32_t key = 0;
    CHECK(nbytes == 0 || nbytes == sizeof(int));
    if (nbytes == sizeof(int)) {
      memcpy(&key, signature, sizeof(int));
      key = key + 1;
    }
    CHECK_LT(key, 100);
    if (kmap_.size() <= key) {
      kmap_.resize(key + 1, nullptr);
    }
    return &(kmap_[key]);
  }

 private:
  std::vector<UopKernel*> kmap_;
};

enum PipelineStage : int {
  kNoneStage = 0,
  kLoadStage = 1,
  kComputeStage = 2,
  kStoreStage = 3
};

// Instruction Queue
template<int kMaxBytes, bool kCoherent, bool kAlwaysCache>
class InsnQueue : public BaseQueue {
 public:
  /*! \brief Initialize the space. */
  void InitSpace() {
    BaseQueue::InitSpace(kElemBytes, kMaxBytes, kCoherent, kAlwaysCache);
    // Initialize the stage
    std::fill(pending_pop_prev_, pending_pop_prev_ + 4, 0);
    std::fill(pending_pop_next_, pending_pop_next_ + 4, 0);
  }
  /*! \return The data pointer. */
  VTAGenericInsn* data() {
    return reinterpret_cast<VTAGenericInsn*>(dram_buffer_);
  }
  /*! \return Number of instructions. */
  uint32_t count() {
    return dram_end_;
  }
  // Insert dependency push of load
  void DepPop(int from, int to) {
    // NOTE: This instruction executes on queue[to]
    if (from < to) {
      if (pending_pop_prev_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_prev_[to] = 1;
    } else {
      if (pending_pop_next_[to]) {
        this->CommitPendingPop(to);
      }
      pending_pop_next_[to] = 1;
    }
    // Impossible condition
    CHECK(from != kLoadStage || to != kStoreStage);
    CHECK(to != kLoadStage || to != kComputeStage);
  }
  // Insert dependency push of load
  void DepPush(int from, int to) {
    // NOTE: this instruction executes on queue[from]
    this->CommitPendingPop(from);
    if (dram_end_ != 0) {
      VTAMemInsn* mptr =
          reinterpret_cast<VTAMemInsn*>(dram_buffer_) + dram_end_ - 1;
      if (GetPipelineStage(mptr) == from) {
        if (from < to && !mptr->push_next_dep) {
          // push(LD->C) or push(C->ST)
          mptr->push_next_dep = true; return;
        } else if (from > to && !mptr->push_prev_dep) {
          // push(C->LD) or push(ST->C)
          mptr->push_prev_dep = true; return;
        }
      }
    }
    if (from < to) {
      // Push next dep
      PushNoop(from, false, true, false, false);
    } else {
      // Push prev dep
      PushNoop(from, true, false, false, false);
    }
  }
  // Create a new instruction for a GEMM stage
  VTAGemInsn* CreateGemInsn() {
    return reinterpret_cast<VTAGemInsn*>(
        Create(kComputeStage));
  }
  // Create a new instruction for a ALU stage
  VTAAluInsn* CreateAluInsn() {
    return reinterpret_cast<VTAAluInsn*>(
        Create(kComputeStage));
  }
  // Create a new instruction for a memory stage
  VTAMemInsn* CreateMemInsn(int memory_type) {
    return reinterpret_cast<VTAMemInsn*>(
        Create(GetMemPipelineStage(memory_type)));
  }
  // create a new instruction for a store stage
  VTAMemInsn* CreateStoreInsn() {
    return reinterpret_cast<VTAMemInsn*>(
        Create(kStoreStage));
  }
  // Rewrite instruction stream to force serial execution
  void RewriteForceSerial() {
    int insn_count = count();
    VTAMemInsn* mem_ptr = reinterpret_cast<VTAMemInsn*>(data());
    for (int i = 1; i < insn_count; ++i) {
      PipelineStage prev = GetPipelineStage(mem_ptr + i - 1);
      PipelineStage now = GetPipelineStage(mem_ptr + i);
      if (prev == kLoadStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else if (prev == kComputeStage && now == kLoadStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kStoreStage && now == kComputeStage) {
        mem_ptr[i - 1].push_prev_dep = true;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = true;
      } else if (prev == kComputeStage && now == kStoreStage) {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = true;
        mem_ptr[i].pop_prev_dep = true;
        mem_ptr[i].pop_next_dep = false;
      } else {
        mem_ptr[i - 1].push_prev_dep = false;
        mem_ptr[i - 1].push_next_dep = false;
        mem_ptr[i].pop_prev_dep = false;
        mem_ptr[i].pop_next_dep = false;
      }
    }
  }

  // Helper function: Get Opcode string
  const char* getOpcodeString(int opcode, bool use_imm) {
      // The string name
      if (opcode == VTA_ALU_OPCODE_MIN) {
          if (use_imm) {
              return "min imm";
          } else {
              return "min";
          }
      } else if (opcode == VTA_ALU_OPCODE_MAX) {
          if (use_imm) {
              return "max imm";
          } else {
              return "max";
          }
      } else if (opcode == VTA_ALU_OPCODE_ADD) {
          if (use_imm) {
              return "add imm";
          } else {
              return "add";
          }
      } else if (opcode == VTA_ALU_OPCODE_SHR) {
          return "shr";
      }

      return "unknown op";
  }

  // Dump instructions in the queue
  void DumpInsn() {
    // Keep tabs on dependence queues
    int l2g_queue = 0;
    int g2l_queue = 0;
    int s2g_queue = 0;
    int g2s_queue = 0;
    // Converter
    union VTAInsn c;
    // Iterate over all instructions
    int insn_count = count();
    const VTAGenericInsn* insn = data();
    printf("There are %u instructions\n", insn_count);
    for (int i = 0; i < insn_count; ++i) {
      // Fetch instruction and decode opcode
      c.generic = insn[i];
      printf("INSTRUCTION %u: ", i);
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.x_size == 0) {
          if (c.mem.opcode == VTA_OPCODE_STORE) {
            printf("NOP-STORE-STAGE\n");
          } else if (GetMemPipelineStage(c.mem.memory_type) == kComputeStage) {
            printf("NOP-COMPUTE-STAGE\n");
          } else {
            printf("NOP-MEMORY-STAGE\n");
          }
          printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                 static_cast<int>(c.mem.pop_prev_dep),
                 static_cast<int>(c.mem.pop_next_dep),
                 static_cast<int>(c.mem.push_prev_dep),
                 static_cast<int>(c.mem.push_next_dep));
          // Count status in queues
          if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
            if (c.mem.opcode == VTA_OPCODE_STORE) {
                CHECK(c.mem.pop_next_dep == false);
                CHECK(c.mem.push_next_dep == false);
                if (c.mem.pop_prev_dep) g2s_queue--;
                if (c.mem.push_prev_dep) s2g_queue++;
            } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                       (c.mem.memory_type == VTA_MEM_ID_INP ||
                        c.mem.memory_type == VTA_MEM_ID_WGT) ) {
                CHECK(c.mem.pop_prev_dep == false);
                CHECK(c.mem.push_prev_dep == false);
                if (c.mem.pop_next_dep) g2l_queue--;
                if (c.mem.push_next_dep) l2g_queue++;
            } else {
                if (c.mem.pop_prev_dep) l2g_queue--;
                if (c.mem.push_prev_dep) g2l_queue++;
                if (c.mem.pop_next_dep) s2g_queue--;
                if (c.mem.push_next_dep) g2s_queue++;
            }
          } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
            // Print instruction field information
            if (c.gemm.pop_prev_dep) l2g_queue--;
            if (c.gemm.push_prev_dep) g2l_queue++;
            if (c.gemm.pop_next_dep) s2g_queue--;
            if (c.gemm.push_next_dep) g2s_queue++;
          }
          printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
          printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
          continue;
        }
        // Print instruction field information
        if (c.mem.opcode == VTA_OPCODE_LOAD) {
          printf("LOAD ");
          if (c.mem.memory_type == VTA_MEM_ID_UOP) printf("UOP\n");
          if (c.mem.memory_type == VTA_MEM_ID_WGT) printf("WGT\n");
          if (c.mem.memory_type == VTA_MEM_ID_INP) printf("INP\n");
          if (c.mem.memory_type == VTA_MEM_ID_ACC) printf("ACC\n");
        }
        if (c.mem.opcode == VTA_OPCODE_STORE) {
          printf("STORE:\n");
        }
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep),
               static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep),
               static_cast<int>(c.mem.push_next_dep));
        printf("\tDRAM: 0x%08x, SRAM:0x%04x\n",
               static_cast<int>(c.mem.dram_base),
               static_cast<int>(c.mem.sram_base));
        printf("\ty: size=%d, pad=[%d, %d]\n",
               static_cast<int>(c.mem.y_size),
               static_cast<int>(c.mem.y_pad_0),
               static_cast<int>(c.mem.y_pad_1));
        printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n",
               static_cast<int>(c.mem.x_size),
               static_cast<int>(c.mem.x_stride),
               static_cast<int>(c.mem.x_pad_0),
               static_cast<int>(c.mem.x_pad_1));
      } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
        // Print instruction field information
        printf("GEMM\n");

        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep),
               static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep),
               static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.gemm.reset_reg));
        printf("\trange (%d, %d)\n",
               static_cast<int>(c.gemm.uop_bgn),
               static_cast<int>(c.gemm.uop_end));
        printf("\touter loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_out),
               static_cast<int>(c.gemm.wgt_factor_out),
               static_cast<int>(c.gemm.src_factor_out),
               static_cast<int>(c.gemm.dst_factor_out));
        printf("\tinner loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
               static_cast<int>(c.gemm.iter_in),
               static_cast<int>(c.gemm.wgt_factor_in),
               static_cast<int>(c.gemm.src_factor_in),
               static_cast<int>(c.gemm.dst_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
               static_cast<int>(c.mem.pop_prev_dep),
               static_cast<int>(c.mem.pop_next_dep),
               static_cast<int>(c.mem.push_prev_dep),
               static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
        printf("\trange (%d, %d)\n",
               static_cast<int>(c.alu.uop_bgn),
               static_cast<int>(c.alu.uop_end));
        printf("\touter loop - iter: %d, dst: %d, src: %d\n",
               static_cast<int>(c.alu.iter_out),
               static_cast<int>(c.alu.dst_factor_out),
               static_cast<int>(c.alu.src_factor_out));
        printf("\tinner loop - iter: %d, dst: %d, src: %d\n",
               static_cast<int>(c.alu.iter_in),
               static_cast<int>(c.alu.dst_factor_in),
               static_cast<int>(c.alu.src_factor_in));
      } else if (c.mem.opcode == VTA_OPCODE_FINISH) {
        printf("FINISH\n");
      }

      // Count status in queues
      if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.opcode == VTA_OPCODE_STORE) {
            CHECK(c.mem.pop_next_dep == false);
            CHECK(c.mem.push_next_dep == false);
            if (c.mem.pop_prev_dep) g2s_queue--;
            if (c.mem.push_prev_dep) s2g_queue++;
        } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                   (c.mem.memory_type == VTA_MEM_ID_INP ||
                    c.mem.memory_type == VTA_MEM_ID_WGT) ) {
            CHECK(c.mem.pop_prev_dep == false);
            CHECK(c.mem.push_prev_dep == false);
            if (c.mem.pop_next_dep) g2l_queue--;
            if (c.mem.push_next_dep) l2g_queue++;
        } else {
            if (c.mem.pop_prev_dep) l2g_queue--;
            if (c.mem.push_prev_dep) g2l_queue++;
            if (c.mem.pop_next_dep) s2g_queue--;
            if (c.mem.push_next_dep) g2s_queue++;
        }
      } else if (c.mem.opcode == VTA_OPCODE_GEMM ||
                 c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        if (c.gemm.pop_prev_dep) l2g_queue--;
        if (c.gemm.push_prev_dep) g2l_queue++;
        if (c.gemm.pop_next_dep) s2g_queue--;
        if (c.gemm.push_next_dep) g2s_queue++;
      }
      printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
      printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
    }
  }

  // Commit all pending pop of corresponding stage
  void CommitPendingPop(int stage) {
    // Handle the LD<->compute queue
    // NOTE: pop executes on target(stage)
    CHECK(stage > 0 && stage < 4);
    if (pending_pop_prev_[stage] ||
        pending_pop_next_[stage]) {
      PushNoop(stage, false, false,
               pending_pop_prev_[stage],
               pending_pop_next_[stage]);
      pending_pop_prev_[stage] = 0;
      pending_pop_next_[stage] = 0;
    }
  }

  void CommitPending() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      CommitPendingPop(i);
    }
  }

  bool PendingPop() {
    for (int i = kLoadStage; i <= kStoreStage; ++i) {
      if (pending_pop_prev_[i]) return true;
      if (pending_pop_next_[i]) return true;
    }
    return false;
  }

 protected:
  /*! \return Add new instruction to the buffer. */
  VTAGenericInsn* NextInsn() {
    VTAGenericInsn* insn  = data() + dram_end_;
    ++dram_end_;
    CHECK(dram_end_ < kMaxElems);
    return insn;
  }
  // Create a new instruction for a given stage
  VTAGenericInsn* Create(PipelineStage stage) {
    VTAGenericInsn* gptr = NextInsn();
    VTAMemInsn* mptr = reinterpret_cast<VTAMemInsn*>(gptr);
    mptr->pop_prev_dep = pending_pop_prev_[stage];
    mptr->pop_next_dep = pending_pop_next_[stage];
    mptr->push_prev_dep = false;
    mptr->push_next_dep = false;
    pending_pop_prev_[stage] = 0;
    pending_pop_next_[stage] = 0;
    return gptr;
  }
  // Get stage of the memory
  static PipelineStage GetMemPipelineStage(int memory_type) {
    if (memory_type == VTA_MEM_ID_ACC) return kComputeStage;
    if (memory_type == VTA_MEM_ID_UOP) return kComputeStage;
    return kLoadStage;
  }
  // Get stage of the computation
  static PipelineStage GetPipelineStage(VTAMemInsn* insn) {
    if (insn->opcode == VTA_OPCODE_GEMM) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_ALU) return kComputeStage;
    if (insn->opcode == VTA_OPCODE_LOAD) {
      if (insn->x_size == 0) return kNoneStage;
      if (insn->memory_type == VTA_MEM_ID_ACC) return kComputeStage;
      if (insn->memory_type == VTA_MEM_ID_UOP) return kComputeStage;
      return kLoadStage;
    }
    if (insn->opcode == VTA_OPCODE_STORE) {
      // FIXME: Right now memory_type is a 2-bit field which means that
      //        VTA_MEM_ID_OUT will appear as 0. For now we'll refrain from
      //        checking the memory_type to avoid an CHECKion error...
      return kStoreStage;
    }
    LOG(FATAL) << "not reached";
    return kNoneStage;
  }
  // Push no-op
  void PushNoop(int stage,
                bool push_prev_dep, bool push_next_dep,
                bool pop_prev_dep, bool pop_next_dep) {
    VTAMemInsn* insn = reinterpret_cast<VTAMemInsn*>(NextInsn());
    insn->opcode = (stage == kStoreStage ? VTA_OPCODE_STORE : VTA_OPCODE_LOAD);
    insn->push_prev_dep = push_prev_dep;
    insn->push_next_dep = push_next_dep;
    insn->pop_prev_dep = pop_prev_dep;
    insn->pop_next_dep = pop_next_dep;
    insn->sram_base = 0;
    insn->dram_base = 0;
    insn->y_size = 0;
    insn->x_size = 0;
    insn->x_stride = 0;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    insn->memory_type = (stage == kLoadStage ? VTA_MEM_ID_INP : VTA_MEM_ID_UOP);
  }

 private:
  // Pending pop of each isntruction queue, qid=0 is not used
  int pending_pop_prev_[4];
  int pending_pop_next_[4];
  static constexpr int kElemBytes = sizeof(VTAGenericInsn);
  static constexpr int kMaxElems = kMaxBytes / kElemBytes;
};

/*!
 * \brief The command queue object that handles the request.
 */
class CommandQueue {
 public:
  CommandQueue() {
    this->InitSpace();
  }
  void InitSpace() {
    uop_queue_.InitSpace();
    insn_queue_.InitSpace();
    device_ = VTADeviceAlloc();
    CHECK(device_ != nullptr);
    printf("Initialize VTACommandHandle...\n");
  }

  ~CommandQueue() {
    VTADeviceFree(device_);
    printf("Close VTACommandhandle...\n");
  }

  uint32_t GetElemBytes(uint32_t memory_id) {
    switch (memory_id) {
      case VTA_MEM_ID_UOP: return VTA_UOP_ELEM_BYTES;
      case VTA_MEM_ID_INP: return VTA_INP_ELEM_BYTES;
      case VTA_MEM_ID_WGT: return VTA_WGT_ELEM_BYTES;
      case VTA_MEM_ID_ACC: return VTA_ACC_ELEM_BYTES;
      case VTA_MEM_ID_OUT: return VTA_INP_ELEM_BYTES;
      default: break;
    }
    LOG(FATAL) << "Memory id not recognized:" << memory_id;
    return 0;
  }

  void LoadBuffer2D(void* src_dram_addr,
                    uint32_t src_elem_offset,
                    uint32_t x_size,
                    uint32_t y_size,
                    uint32_t x_stride,
                    uint32_t x_pad_before,
                    uint32_t y_pad_before,
                    uint32_t x_pad_after,
                    uint32_t y_pad_after,
                    uint32_t dst_sram_index,
                    uint32_t dst_memory_type) {
    VTAMemInsn* insn = insn_queue_.CreateMemInsn(dst_memory_type);
    insn->opcode = VTA_OPCODE_LOAD;
    insn->memory_type = dst_memory_type;
    insn->sram_base = dst_sram_index;
    DataBuffer* src = DataBuffer::FromHandle(src_dram_addr);
    insn->dram_base = src->phy_addr() / GetElemBytes(dst_memory_type) + src_elem_offset;
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = y_pad_before;
    insn->y_pad_1 = y_pad_after;
    insn->x_pad_0 = x_pad_before;
    insn->x_pad_1 = x_pad_after;
    this->CheckInsnOverFlow();
  }

  void StoreBuffer2D(uint32_t src_sram_index,
                     uint32_t src_memory_type,
                     void* dst_dram_addr,
                     uint32_t dst_elem_offset,
                     uint32_t x_size,
                     uint32_t y_size,
                     uint32_t x_stride) {
    VTAMemInsn* insn = insn_queue_.CreateStoreInsn();
    insn->opcode = VTA_OPCODE_STORE;
    insn->memory_type = src_memory_type;
    insn->sram_base = src_sram_index;
    DataBuffer* dst = DataBuffer::FromHandle(dst_dram_addr);
    insn->dram_base = dst->phy_addr() / GetElemBytes(src_memory_type) + dst_elem_offset;
    insn->y_size = y_size;
    insn->x_size = x_size;
    insn->x_stride = x_stride;
    insn->y_pad_0 = 0;
    insn->y_pad_1 = 0;
    insn->x_pad_0 = 0;
    insn->x_pad_1 = 0;
    this->CheckInsnOverFlow();
  }

  void DepPush(int from_qid, int to_qid) {
    insn_queue_.DepPush(from_qid, to_qid);
  }

  void DepPop(int from_qid, int to_qid) {
    insn_queue_.DepPop(from_qid, to_qid);
  }

  void ReadBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (!(debug_flag_ & VTA_DEBUG_SKIP_READ_BARRIER)) {
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      DataBuffer::FromHandle(buffer)->FlushCache(
          elem_bytes * start, elem_bytes * extent);
    }
  }

  void WriteBarrier(void* buffer, uint32_t elem_bits, uint32_t start, uint32_t extent) {
    if (!(debug_flag_ & VTA_DEBUG_SKIP_WRITE_BARRIER)) {
      uint32_t elem_bytes = (elem_bits + 8 - 1) / 8;
      DataBuffer::FromHandle(buffer)->InvalidateCache(
          elem_bytes * start, elem_bytes * extent);
    }
  }

  void Synchronize(uint32_t wait_cycles) {
    // Insert dependences to force serialization
    if (debug_flag_ & VTA_DEBUG_FORCE_SERIAL) {
      insn_queue_.RewriteForceSerial();
    }
    // This will issue finish after last store finishes
    insn_queue_.DepPush(kStoreStage, kComputeStage);
    insn_queue_.DepPush(kLoadStage, kComputeStage);
    insn_queue_.DepPop(kStoreStage, kComputeStage);
    insn_queue_.DepPop(kLoadStage, kComputeStage);
    insn_queue_.CommitPendingPop(kComputeStage);
    // NOTE: FINISH cannot contain pop
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_FINISH;
    CHECK(!insn_queue_.PendingPop());
    // Check if there are no instruction to execute at all
    if (insn_queue_.count() == 0) return;
    // Synchronization for the queues
    uop_queue_.AutoReadBarrier();
    insn_queue_.AutoReadBarrier();
    // Dump instructions if debug enabled
    if (debug_flag_ & VTA_DEBUG_DUMP_INSN) {
      insn_queue_.DumpInsn();
    }
    // Make sure that the last instruction is a finish instruction
    CHECK(reinterpret_cast<VTAMemInsn*>(
        insn_queue_.data())[insn_queue_.count()-1].opcode == VTA_OPCODE_FINISH);

    // Make sure that we don't exceed contiguous physical memory limits
    CHECK(insn_queue_.count() * sizeof(VTAGenericInsn) < VTA_MAX_XFER);
    int timeout = VTADeviceRun(
        device_,
        insn_queue_.dram_phy_addr(),
        insn_queue_.count(),
        wait_cycles);
    CHECK_EQ(timeout, 0);
    // Reset buffers
    uop_queue_.Reset();
    insn_queue_.Reset();
  }

  // Get record kernel
  UopKernel* record_kernel() const {
    CHECK(record_kernel_ != nullptr);
    return record_kernel_;
  }

  // Set debug flag
  void SetDebugFlag(int debug_flag) {
    debug_flag_ = debug_flag;
  }

  void PushGEMMOp(void** uop_handle,
                  int (*finit)(void*),
                  void* signature,
                  int nbytes) {
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);
      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushGEMMOp(static_cast<UopKernel*>(kptr[0]));
    this->CheckInsnOverFlow();
  }

  void PushALUUop(void** uop_handle,
                  int (*finit)(void*),
                  void* signature,
                  int nbytes) {
    UopKernelMap** uptr = reinterpret_cast<UopKernelMap**>(uop_handle);
    if (uptr[0] == nullptr) {
      uptr[0] = new UopKernelMap();
    }
    UopKernel** kptr = uptr[0]->Get(signature, nbytes);
    if (kptr[0] == nullptr) {
      record_kernel_ = new UopKernel(static_cast<char*>(signature), nbytes);
      CHECK_EQ((*finit)(signature), 0);
      kptr[0] = static_cast<UopKernel*>(record_kernel_);
      if (debug_flag_ & VTA_DEBUG_DUMP_UOP) {
        record_kernel_->Dump();
      }
      record_kernel_ = nullptr;
    }
    this->PushALUUop(static_cast<UopKernel*>(kptr[0]));
    this->CheckInsnOverFlow();
  }

  static std::shared_ptr<CommandQueue>& ThreadLocal() {
    static std::shared_ptr<CommandQueue> inst =
        std::make_shared<CommandQueue>();
    if (inst == nullptr) {
      inst = std::make_shared<CommandQueue>();
    }
    return inst;
  }

  static void Shutdown() {
    ThreadLocal().reset();
  }

 private:
  // Push GEMM uop to the command buffer
  void PushGEMMOp(UopKernel* kernel) {
    uop_queue_.Push(kernel,
                    [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);
    }
    VTAGemInsn* insn = insn_queue_.CreateGemInsn();
    insn->opcode = VTA_OPCODE_GEMM;
    insn->reset_reg = kernel->reset_out_;
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    const std::vector<UopKernel::LoopEntry> &loop = kernel->loop();
    if (loop.size() > 0) {
      insn->iter_out = loop[0].extent;
      insn->wgt_factor_out = loop[0].wgt_factor;
      insn->src_factor_out = loop[0].src_factor;
      insn->dst_factor_out = loop[0].dst_factor;
    } else {
      insn->iter_out = 1;
      insn->wgt_factor_out = 0;
      insn->src_factor_out = 0;
      insn->dst_factor_out = 0;
    }
    if (loop.size() > 1) {
      insn->iter_in = loop[1].extent;
      insn->wgt_factor_in = loop[1].wgt_factor;
      insn->src_factor_in = loop[1].src_factor;
      insn->dst_factor_in = loop[1].dst_factor;
    } else {
      insn->iter_in = 1;
      insn->wgt_factor_in = 0;
      insn->src_factor_in = 0;
      insn->dst_factor_in = 0;
    }
  }

  // Push ALU uop to the command buffer
  void PushALUUop(UopKernel* kernel) {
    uop_queue_.Push(kernel,
                    [this]() { this->AutoSync(); });
    if (uop_queue_.pending()) {
      VTAMemInsn* insn = insn_queue_.CreateMemInsn(VTA_MEM_ID_UOP);
      insn->opcode = VTA_OPCODE_LOAD;
      uop_queue_.FlushUopLoad(insn);
    }
    VTAAluInsn* insn = insn_queue_.CreateAluInsn();
    insn->opcode = VTA_OPCODE_ALU;
    insn->reset_reg = kernel->reset_out_;
    insn->uop_bgn = kernel->sram_begin_;
    insn->uop_end = kernel->sram_end_;
    insn->alu_opcode = kernel->opcode_;
    insn->use_imm = kernel->use_imm_;
    insn->imm = kernel->imm_val_;
    const std::vector<UopKernel::LoopEntry> &loop = kernel->loop();
    if (loop.size() == 0) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      insn->iter_in = 1;
      insn->dst_factor_in = 0;
      insn->src_factor_in = 0;
    } else if (loop.size() == 1) {
      insn->iter_out = 1;
      insn->dst_factor_out = 0;
      insn->src_factor_out = 0;
      insn->iter_in = loop[0].extent;
      insn->dst_factor_in = loop[0].dst_factor;
      insn->src_factor_in = loop[0].src_factor;
    } else {
      insn->iter_out = loop[0].extent;
      insn->dst_factor_out = loop[0].dst_factor;
      insn->src_factor_out = loop[0].src_factor;
      insn->iter_in = loop[1].extent;
      insn->dst_factor_in = loop[1].dst_factor;
      insn->src_factor_in = loop[1].src_factor;
    }
  }

  void CheckInsnOverFlow() {
    // At each API call, we can at most commit:
    // one pending store, one pending load, and one uop
    if ((insn_queue_.count() + 4) * sizeof(VTAGenericInsn) >= VTA_MAX_XFER) {
      this->AutoSync();
    }
  }
  // Auto sync when instruction overflow
  void AutoSync() {
    this->Synchronize(1 << 31);
  }

  // Internal debug flag
  int debug_flag_{0};
  // The kernel we currently recording
  UopKernel* record_kernel_{nullptr};
  // Micro op queue
  UopQueue<VTA_MAX_XFER, true, true> uop_queue_;
  // instruction queue
  InsnQueue<VTA_MAX_XFER, true, true> insn_queue_;
  // Device handle
  VTADeviceHandle device_{nullptr};
};

}  // namespace vta

void* VTABufferAlloc(size_t size) {
  return vta::DataBuffer::Alloc(size);
}

void VTABufferFree(void* buffer) {
  vta::DataBuffer::Free(vta::DataBuffer::FromHandle(buffer));
}

void VTABufferCopy(const void* from,
                   size_t from_offset,
                   void* to,
                   size_t to_offset,
                   size_t size,
                   int kind_mask) {
  vta::DataBuffer* from_buffer = nullptr;
  vta::DataBuffer* to_buffer = nullptr;

  if (kind_mask & 2) {
    from_buffer = vta::DataBuffer::FromHandle(from);
    from = from_buffer->virt_addr();
  }
  if (kind_mask & 1) {
    to_buffer = vta::DataBuffer::FromHandle(to);
    to = to_buffer->virt_addr();
  }
  if (from_buffer) {
    from_buffer->InvalidateCache(from_offset, size);
  }

  memcpy(static_cast<char*>(to) + to_offset,
         static_cast<const char*>(from) + from_offset,
         size);
  if (to_buffer) {
    to_buffer->FlushCache(to_offset, size);
  }
}

VTACommandHandle VTATLSCommandHandle() {
  return vta::CommandQueue::ThreadLocal().get();
}

void VTARuntimeShutdown() {
  vta::CommandQueue::Shutdown();
}

void VTASetDebugMode(VTACommandHandle cmd, int debug_flag) {
  static_cast<vta::CommandQueue*>(cmd)->
      SetDebugFlag(debug_flag);
}

void* VTABufferCPUPtr(VTACommandHandle cmd, void* buffer) {
  return vta::DataBuffer::FromHandle(buffer)->virt_addr();
}

void VTAWriteBarrier(VTACommandHandle cmd,
                     void* buffer,
                     uint32_t elem_bits,
                     uint32_t start,
                     uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->
      WriteBarrier(buffer, elem_bits, start, extent);
}

void VTAReadBarrier(VTACommandHandle cmd,
                    void* buffer,
                    uint32_t elem_bits,
                    uint32_t start,
                    uint32_t extent) {
  static_cast<vta::CommandQueue*>(cmd)->
      ReadBarrier(buffer, elem_bits, start, extent);
}

void VTALoadBuffer2D(VTACommandHandle cmd,
                     void* src_dram_addr,
                     uint32_t src_elem_offset,
                     uint32_t x_size,
                     uint32_t y_size,
                     uint32_t x_stride,
                     uint32_t x_pad_before,
                     uint32_t y_pad_before,
                     uint32_t x_pad_after,
                     uint32_t y_pad_after,
                     uint32_t dst_sram_index,
                     uint32_t dst_memory_type) {
  static_cast<vta::CommandQueue*>(cmd)->
      LoadBuffer2D(src_dram_addr, src_elem_offset,
                   x_size, y_size, x_stride,
                   x_pad_before, y_pad_before,
                   x_pad_after, y_pad_after,
                   dst_sram_index, dst_memory_type);
}

void VTAStoreBuffer2D(VTACommandHandle cmd,
                      uint32_t src_sram_index,
                      uint32_t src_memory_type,
                      void* dst_dram_addr,
                      uint32_t dst_elem_offset,
                      uint32_t x_size,
                      uint32_t y_size,
                      uint32_t x_stride) {
  static_cast<vta::CommandQueue*>(cmd)->
      StoreBuffer2D(src_sram_index, src_memory_type,
                    dst_dram_addr, dst_elem_offset,
                    x_size, y_size, x_stride);
}

void VTAUopPush(uint32_t mode,
                uint32_t reset_out,
                uint32_t dst_index,
                uint32_t src_index,
                uint32_t wgt_index,
                uint32_t opcode,
                uint32_t use_imm,
                int32_t imm_val) {
  vta::CommandQueue::ThreadLocal()->record_kernel()
      ->Push(mode, reset_out, dst_index, src_index,
             wgt_index, opcode, use_imm, imm_val);
}

void VTAUopLoopBegin(uint32_t extent,
                     uint32_t dst_factor,
                     uint32_t src_factor,
                     uint32_t wgt_factor) {
  vta::CommandQueue::ThreadLocal()->record_kernel()
      ->PushLoopBegin(extent, dst_factor, src_factor, wgt_factor);
}

void VTAUopLoopEnd() {
  vta::CommandQueue::ThreadLocal()->record_kernel()
      ->PushLoopEnd();
}

int VTAPushGEMMOp(void** uop_handle,
                  int (*finit)(void*),
                  void* signature,
                  int nbytes) {
  vta::CommandQueue::ThreadLocal()->
      PushGEMMOp(uop_handle, finit, signature, nbytes);
  return 0;
}

int VTAPushALUOp(void** uop_handle,
                 int (*finit)(void*),
                 void* signature,
                 int nbytes) {
  vta::CommandQueue::ThreadLocal()->
      PushALUUop(uop_handle, finit, signature, nbytes);
  return 0;
}

int VTADepPush(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->
      DepPush(from_qid, to_qid);
  return 0;
}

int VTADepPop(VTACommandHandle cmd, int from_qid, int to_qid) {
  static_cast<vta::CommandQueue*>(cmd)->
      DepPop(from_qid, to_qid);
  return 0;
}

void VTASynchronize(VTACommandHandle cmd, uint32_t wait_cycles) {
  static_cast<vta::CommandQueue*>(cmd)->
      Synchronize(wait_cycles);
}
