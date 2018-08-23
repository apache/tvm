/*!
 *  Copyright (c) 2017 by Contributors
 * \file vpi_device.cc
 * \brief Simulated VPI RAM device.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/packed_func_ext.h>
#include <cstdlib>
#include <unordered_map>
#include <map>
#include <queue>
#include "vpi_session.h"

namespace tvm {
namespace codegen {

/*! \brief Simulated device ram */
class VPIDeviceAPI final : public runtime::DeviceAPI {
 public:
  VPIDeviceAPI() {
    const char* s_ram_size = getenv("TVM_VPI_RAM_SIZE_MB");
    // 16 MB ram.
    int ram_size = 32;
    if (s_ram_size != nullptr) {
      ram_size = atoi(s_ram_size);
    }
    ram_.resize(ram_size << 17);
    ram_head_ = runtime::kAllocAlignment;
    ram_max_ = ram_.size() * sizeof(int64_t);
    LOG(INFO) << "Initialize VPI simulated ram " << ram_size << "MB ...";
  }
  // convert address to real address
  void* RealAddr(const void* addr, size_t size) const {
    int64_t ptr = reinterpret_cast<int64_t>(addr);
    CHECK_LE(ptr + size, ram_max_)
        << "VPI: Illegal memory access";
    return (char*)(&ram_[0]) + ptr;  // NOLINT(*)
  }
  // convert address to real address
  void* RealAddrSafe(const void* addr, size_t size) const {
    int64_t ptr = reinterpret_cast<int64_t>(addr);
    if (ptr + size >= ram_max_) return nullptr;
    return (char*)(&ram_[0]) + ptr;  // NOLINT(*)
  }
  void SetDevice(TVMContext ctx) final {}
  void GetAttr(TVMContext ctx, runtime::DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == runtime::kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(TVMContext ctx,
                       size_t size,
                       size_t alignment,
                       TVMType type_hint) final {
    // always align to 32 bytes at least.
    CHECK_LE(alignment, runtime::kAllocAlignment);
    alignment = runtime::kAllocAlignment;
    // always allocate block with aligned size.
    size += alignment - (size % alignment);
    // This is not thread safe, but fine for simulation.
    auto it = free_blocks_.lower_bound(size);
    if (it != free_blocks_.end()) {
      size_t head = it->second;
      free_blocks_.erase(it);
      Block& b = block_map_.at(head);
      CHECK(b.is_free);
      b.is_free = false;
      return reinterpret_cast<void*>(head);
    } else {
      CHECK_EQ(ram_head_ % runtime::kAllocAlignment, 0U);
      Block b;
      b.size = size;
      b.is_free = false;
      CHECK_LE(ram_head_ + size, ram_max_)
          << "VPI: Out of memory";
      block_map_[ram_head_] = b;
      void* ret = reinterpret_cast<void*>(ram_head_);
      ram_head_ += size;
      return ret;
    }
  }
  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    size_t head = reinterpret_cast<size_t>(ptr);
    Block& b = block_map_.at(head);
    b.is_free = true;
    free_blocks_.insert({b.size, head});
  }
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
    if (static_cast<int>(ctx_from.device_type) == kDLVPI) {
      from = RealAddr(static_cast<const char*>(from) + from_offset, size);
    }
    if (static_cast<int>(ctx_to.device_type) == kDLVPI) {
      to = RealAddr(static_cast<char*>(to) + to_offset, size);
    }
    memcpy(to, from, size);
  }
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }
  static VPIDeviceAPI* Global() {
    static VPIDeviceAPI inst;
    return &inst;
  }

 private:
  // allocator block for reuse
  struct Block {
    // The size of the block
    size_t size;
    // Whether this is already freed.
    bool is_free{true};
  };
  // head -> blocks
  std::unordered_map<size_t, Block> block_map_;
  // size -> free heads.
  std::multimap<size_t, size_t> free_blocks_;
  // top of the ram
  size_t ram_head_, ram_max_;
  // The ram space.
  std::vector<int64_t> ram_;
};

/* !\brief vector buffer to help read/write */
class VPIVecBuffer {
 public:
  // Put data into vec
  void put_vec(const VPIHandle& h, size_t nwords,
               const void* dptr, size_t size) {
    wbuf_.resize(nwords);
    vbuf_.resize(nwords);
    memcpy(&wbuf_[0], dptr, size);
    for (size_t i = 0; i < nwords; ++i) {
      vbuf_[i].aval = wbuf_[i];
      vbuf_[i].bval = 0;
    }
    h.put_vec(vbuf_);
  }
  // read data from vec.
  void get_vec(const VPIHandle& h, void* dptr, size_t size) {
    h.get_vec(&vbuf_);
    wbuf_.resize(vbuf_.size());
    for (size_t i = 0; i < vbuf_.size(); ++i) {
      wbuf_[i] = vbuf_[i].aval;
      CHECK_EQ(vbuf_[i].bval, 0)
          << "Write indetermined value to RAM";
    }
    memcpy(dptr, &wbuf_[0], size);
  }

 private:
  // Temporal buffers.
  std::vector<int32_t> wbuf_;
  std::vector<vpi::VPIVecVal> vbuf_;
};

/*!
 * \brief Memory interface for VPI memory.
 */
class VPIMemoryInterface {
 public:
  // Initialize the FSM.
  void Init(VPIHandle module) {
    device_ = VPIDeviceAPI::Global();
    in_rst_ = module["rst"];
    // read ports
    in_read_dequeue_ = module["read_en"];
    out_reg_read_data_ = module["reg_read_data"];
    // Write ports
    in_write_enqueue_ = module["write_en"];
    in_write_data_ = module["write_data_in"];
    // Status port
    out_reg_read_valid_ = module["reg_read_valid"];
    out_reg_write_ready_ = module["reg_write_ready"];
    // memory control signal
    ctrl_read_req_ = module["host_read_req"];
    ctrl_read_addr_ = module["host_read_addr"];
    ctrl_read_size_ = module["host_read_size"];
    ctrl_write_req_ = module["host_write_req"];
    ctrl_write_addr_ = module["host_write_addr"];
    ctrl_write_size_ = module["host_write_size"];
    // The bit and bytes;
    size_t read_bits =  out_reg_read_data_.size();
    size_t write_bits =  in_write_data_.size();
    CHECK_EQ(read_bits % 8U, 0)
        << "Read/write unit have to be multiple of 8 bit(bytes)";
    CHECK_EQ(write_bits % 8U, 0)
        << "Read/write unit have to be multiple of 8 bit(bytes)";
    read_unit_bytes_ = read_bits / 8U;
    write_unit_bytes_ = write_bits / 8U;
  }
  // Callback at neg-edge.
  void AtNegEdge() {
    // reset
    if (in_rst_.get_int()) {
      CHECK_EQ(pending_read_.size, 0U);
      CHECK_EQ(pending_write_.size, 0U);
      CHECK(read_tasks_.empty());
      CHECK(write_tasks_.empty());
      out_reg_write_ready_.put_int(0);
      out_reg_read_valid_.put_int(0);
      return;
    }
    // read write tasks
    if (in_read_dequeue_.get_int() || !out_reg_read_valid_.get_int()) {
      ReadFromFIFO();
    }
    // update write full
    if (in_write_enqueue_.get_int()) {
      CHECK(out_reg_write_ready_.get_int());
      WriteToFIFO();
    }
    if (pending_write_.size || write_tasks_.size()) {
      out_reg_write_ready_.put_int(1);
    } else {
      out_reg_write_ready_.put_int(0);
    }
    // Control tasks
    if (ctrl_read_req_.get_int()) {
      FIFOTask tsk;
      tsk.addr = reinterpret_cast<char*>(ctrl_read_addr_.get_int());
      tsk.size = static_cast<size_t>(ctrl_read_size_.get_int());
      read_tasks_.push(tsk);
    }
    // Control tasks
    if (ctrl_write_req_.get_int()) {
      FIFOTask tsk;
      tsk.addr = reinterpret_cast<char*>(ctrl_write_addr_.get_int());
      tsk.size = static_cast<size_t>(ctrl_write_size_.get_int());
      write_tasks_.push(tsk);
    }
  }

 private:
  // The FIFO tasks
  struct FIFOTask {
    char* addr{nullptr};
    size_t size{0};
  };
  // handle dequeue event
  void ReadFromFIFO() {
    if (pending_read_.size == 0) {
      if (!read_tasks_.empty()) {
        pending_read_ = read_tasks_.front();
        read_tasks_.pop();
        // translate to real memory addr
        pending_read_.addr = static_cast<char*>(
            device_->RealAddr(
                pending_read_.addr, pending_read_.size));
      }
    }
    if (pending_read_.size != 0) {
      // The size to be read
      size_t nread = std::min(pending_read_.size, read_unit_bytes_);
      // Read from the data
      size_t nwords = (read_unit_bytes_ + 3) / 4;
      vbuf_.put_vec(out_reg_read_data_, nwords,
                    pending_read_.addr, nread);
      // Update the pointer
      pending_read_.size -= nread;
      pending_read_.addr += nread;
      // read into the vector
      out_reg_read_valid_.put_int(1);
    } else {
      out_reg_read_valid_.put_int(0);
    }
  }
  // handle write event
  void WriteToFIFO() {
    if (pending_write_.size == 0) {
      if (!write_tasks_.empty()) {
        pending_write_ = write_tasks_.front();
        write_tasks_.pop();
        // translate to real memory addr
        pending_write_.addr = static_cast<char*>(
            device_->RealAddr(
                pending_write_.addr, pending_write_.size));
      }
    }
    if (pending_write_.size != 0) {
      // write to the ram.
      size_t nwrite = std::min(pending_write_.size, write_unit_bytes_);
      vbuf_.get_vec(in_write_data_, pending_write_.addr, nwrite);
      // Update the pointer
      pending_write_.size -= nwrite;
      pending_write_.addr += nwrite;
    }
  }
  // Device API
  VPIDeviceAPI* device_{nullptr};
  // Input clock and reset
  VPIHandle in_rst_;
  // Read FIFO signal
  VPIHandle in_read_dequeue_;
  // Write FIFO signal
  VPIHandle in_write_enqueue_;
  VPIHandle in_write_data_;
  // Read memory controler signals
  VPIHandle ctrl_read_req_;
  VPIHandle ctrl_read_addr_;
  VPIHandle ctrl_read_size_;
  // Write memory controler signal signals
  VPIHandle ctrl_write_req_;
  VPIHandle ctrl_write_addr_;
  VPIHandle ctrl_write_size_;
  // Read FIFO outputs
  VPIHandle out_reg_read_data_;
  VPIHandle out_reg_read_valid_;
  // Write FIFO outputs
  VPIHandle out_reg_write_ready_;
  // Size of current pending read.
  FIFOTask pending_read_;
  FIFOTask pending_write_;
  // The read/write task queues.
  std::queue<FIFOTask> read_tasks_;
  std::queue<FIFOTask> write_tasks_;
  // Unit bytes for read/writing
  size_t read_unit_bytes_;
  size_t write_unit_bytes_;
  // Temporal buffers.
  VPIVecBuffer vbuf_;
};

// Read only memory map.
class VPIMemMapBase {
 public:
  // Initialize the FSM.
  void Init(VPIHandle module, const std::string& data_port) {
    device_ = VPIDeviceAPI::Global();
    // intiatialize the connections
    rst_ = module["rst"];
    addr_ = module["addr"];
    data_ = module[data_port];
    mmap_addr_ = module["mmap_addr"];
    size_t unit_bits =  data_.size();
    CHECK_EQ(unit_bits % 8U, 0)
        << "Read/write unit have to be multiple of 8 bit(bytes)";
    unit_bytes_ = unit_bits / 8U;
  }
  void* RealAddr() {
    int byte_offset = addr_.get_int() * unit_bytes_;
    void* ptr =
        device_->RealAddrSafe(
            reinterpret_cast<void*>(mmap_addr_.get_int() + byte_offset), 1);
    return ptr;
  }

 protected:
  // Device API
  VPIDeviceAPI* device_{nullptr};
  VPIHandle rst_;
  VPIHandle addr_;
  VPIHandle data_;
  VPIHandle mmap_addr_;
  size_t unit_bytes_;
  VPIVecBuffer vbuf_;
};

class VPIReadMemMap : public VPIMemMapBase {
 public:
  void Init(VPIHandle module) {
    VPIMemMapBase::Init(module, "reg_data");
  }
  void AtNegEdge() {
    void* ptr = RealAddr();
    if (ptr == nullptr) return;
    size_t nwords = (unit_bytes_ + 3) / 4;
    vbuf_.put_vec(data_, nwords, ptr, unit_bytes_);
  }
};

// Write only memory map.
class VPIWriteMemMap : public VPIMemMapBase {
 public:
  void Init(VPIHandle module) {
    VPIMemMapBase::Init(module, "data_in");
    enable_ = module["en"];
  }
  void AtNegEdge() {
    if (!enable_.get_int() || rst_.get_int()) return;
    void* ptr = RealAddr();
    CHECK(ptr != nullptr)
        << "Illegal write to VPI RAM";
    vbuf_.get_vec(data_, ptr, unit_bytes_);
  }

 private:
  VPIHandle enable_;
};

TVM_REGISTER_GLOBAL("device_api.vpi")
.set_body([](runtime::TVMArgs args, runtime::TVMRetValue* rv) {
    runtime::DeviceAPI* ptr = VPIDeviceAPI::Global();
    *rv = static_cast<void*>(ptr);
  });

template<typename T>
void TVMVPIHook(runtime::TVMArgs args, runtime::TVMRetValue* rv) {
  VPIHandle m = args[0];
  std::shared_ptr<T> p = std::make_shared<T>();
  p->Init(m);
  LOG(INFO) << "Hook " << m.name() << " to tvm vpi simulation...";
  PackedFunc pf([p](const runtime::TVMArgs&, runtime::TVMRetValue*) {
      p->AtNegEdge();
    });
  *rv = pf;
}

TVM_REGISTER_GLOBAL("_vpi_module_tvm_vpi_mem_interface")
.set_body(TVMVPIHook<VPIMemoryInterface>);

TVM_REGISTER_GLOBAL("_vpi_module_tvm_vpi_read_mmap")
.set_body(TVMVPIHook<VPIReadMemMap>);

TVM_REGISTER_GLOBAL("_vpi_module_tvm_vpi_write_mmap")
.set_body(TVMVPIHook<VPIWriteMemMap>);

}  // namespace codegen
}  // namespace tvm
