/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_session.cc
 * \brief RPC session for remote function call.
 */
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <array>
#include "./rpc_session.h"
#include "../device_api.h"

namespace tvm {
namespace runtime {
// Temp buffer for data array
struct RPCByteArrayBuffer {
  TVMByteArray arr;
  std::string data;
};
// Temp buffer for data array
struct RPCDataArrayBuffer {
  DLTensor tensor;
  std::vector<int64_t> shape;
};
/*!
 * \brief Temporal argument buffer.
 */
struct RPCArgBuffer {
  // The argument values
  std::vector<TVMValue> value;
  // The type codes.
  std::vector<int> tcode;
  // Temporal resources.
  std::vector<std::unique_ptr<RPCByteArrayBuffer> > temp_bytes;
  // Temporal array
  std::vector<std::unique_ptr<RPCDataArrayBuffer> > temp_array;
  // convert buffer as TVMArgs
  TVMArgs AsTVMArgs() const {
    return TVMArgs(value.data(), tcode.data(), value.size());
  }
};

struct RPCSessTable {
 public:
  static constexpr int kMaxRPCSession = 32;
  // Get global singleton
  static RPCSessTable* Global() {
    static RPCSessTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<RPCSession> Get(int index) {
    CHECK(index >= 0 && index < kMaxRPCSession);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<RPCSession> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxRPCSession; ++i) {
      if (tbl_[i].lock() == nullptr) {
        tbl_[i] = ptr; return i;
      }
    }
    LOG(FATAL) << "maximum number of RPC session reached";
    return 0;
  }

 private:
  // The mutex
  std::mutex mutex_;
  // Use weak_ptr intentionally
  // If the RPCSession get released, the pointer session will be released
  std::array<std::weak_ptr<RPCSession>, kMaxRPCSession> tbl_;
};

void RPCSession::Init() {
  // Quick function to call remote.
  call_remote_ = PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
      this->SendPackedSeq(args.values, args.type_codes, args.num_args);
      RPCCode code = RPCCode::kCallFunc;
      while (code != RPCCode::kReturn) {
        code = HandleNextEvent(rv);
      }
    });
}

std::shared_ptr<RPCSession> RPCSession::Create(common::TCPSocket sock) {
  std::shared_ptr<RPCSession> sess = std::make_shared<RPCSession>();
  sess->sock_ = sock;
  sess->Init();
  sess->table_index_ =  RPCSessTable::Global()->Insert(sess);
  return sess;
}

std::shared_ptr<RPCSession> RPCSession::Get(int table_index) {
  return RPCSessTable::Global()->Get(table_index);
}

RPCSession::~RPCSession() {
  this->Shutdown();
}

void RPCSession::Shutdown() {
  if (!sock_.BadSocket()) {
    RPCCode code = RPCCode::kShutdown;
    CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
    sock_.Close();
  }
}

void RPCSession::ServerLoop() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  RPCCode code = RPCCode::kCallFunc;
  TVMRetValue rv;
  while (code != RPCCode::kShutdown) {
    code = HandleNextEvent(&rv);
    CHECK(code != RPCCode::kReturn);
  }
  if (!sock_.BadSocket()) {
    sock_.Close();
  }
}

// Get remote function with name
void RPCSession::CallFunc(void* h, TVMArgs args, TVMRetValue* rv) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  RPCCode code = RPCCode::kCallFunc;
  CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
  uint64_t handle = reinterpret_cast<uint64_t>(h);
  CHECK_EQ(sock_.SendAll(&handle, sizeof(handle)), sizeof(handle));
  call_remote_.CallPacked(args, rv);
}

void RPCSession::CopyToRemote(void* from,
                              size_t from_offset,
                              void* to,
                              size_t to_offset,
                              size_t data_size,
                              TVMContext ctx_to) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  ctx_to = StripSessMask(ctx_to);
  RPCCode code = RPCCode::kCopyToRemote;
  CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
  uint64_t handle = reinterpret_cast<uint64_t>(to);
  CHECK_EQ(sock_.SendAll(&handle, sizeof(handle)), sizeof(handle));
  uint64_t offset = static_cast<uint64_t>(to_offset);
  CHECK_EQ(sock_.SendAll(&offset, sizeof(offset)), sizeof(offset));
  uint64_t size = static_cast<uint64_t>(data_size);
  CHECK_EQ(sock_.SendAll(&size, sizeof(size)), sizeof(size));
  CHECK_EQ(sock_.SendAll(&ctx_to, sizeof(ctx_to)), sizeof(ctx_to));
  CHECK_EQ(sock_.SendAll(reinterpret_cast<char*>(from) + from_offset, data_size),
           data_size);
  TVMRetValue rv;
  while (code != RPCCode::kReturn) {
    code = HandleNextEvent(&rv);
  }
}

void RPCSession::CopyFromRemote(void* from,
                                size_t from_offset,
                                void* to,
                                size_t to_offset,
                                size_t data_size,
                                TVMContext ctx_from) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  ctx_from = StripSessMask(ctx_from);
  RPCCode code = RPCCode::kCopyFromRemote;
  CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
  uint64_t handle = reinterpret_cast<uint64_t>(from);
  CHECK_EQ(sock_.SendAll(&handle, sizeof(handle)), sizeof(handle));
  uint64_t offset = static_cast<uint64_t>(from_offset);
  CHECK_EQ(sock_.SendAll(&offset, sizeof(offset)), sizeof(offset));
  uint64_t size = static_cast<uint64_t>(data_size);
  CHECK_EQ(sock_.SendAll(&size, sizeof(size)), sizeof(size));
  CHECK_EQ(sock_.SendAll(&ctx_from, sizeof(ctx_from)), sizeof(ctx_from));
  CHECK_EQ(sock_.RecvAll(&code, sizeof(code)), sizeof(code));
  if (code == RPCCode::kCopyAck) {
    CHECK_EQ(sock_.RecvAll(reinterpret_cast<char*>(to) + to_offset, data_size),
             data_size);
  } else {
    HandleException();
  }
}

void RPCSession::SendReturnValue(
    int succ, TVMValue ret_value, int ret_tcode) {
  if (succ == 0) {
    RPCCode code = RPCCode::kReturn;
    CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
  } else {
    RPCCode code = RPCCode::kException;
    CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
    ret_value.v_str = TVMGetLastError();
    ret_tcode = kStr;
  }
  SendPackedSeq(&ret_value, &ret_tcode, 1);
}

template<typename F>
void RPCSession::CallHandler(F f) {
  RPCArgBuffer args;
  this->RecvPackedSeq(&args);
  TVMRetValue rv;
  TVMValue ret_value;
  int ret_tcode;
  try {
    f(TVMArgs(args.value.data(), args.tcode.data(),
              static_cast<int>(args.value.size())), &rv);
    RPCCode code = RPCCode::kReturn;
    CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
    if (rv.type_code() == kStr) {
      std::string str = rv;
      ret_value.v_str = str.c_str();
      ret_tcode = kStr;
      SendPackedSeq(&ret_value, &ret_tcode, 1);
    } else {
      ret_value = rv.value();
      ret_tcode = rv.type_code();
      SendPackedSeq(&ret_value, &ret_tcode, 1);
    }
  } catch (const std::runtime_error& e) {
    RPCCode code = RPCCode::kException;
    CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
    ret_value.v_str = e.what();
    ret_tcode = kStr;
    SendPackedSeq(&ret_value, &ret_tcode, 1);
  }
}

void RPCSession::HandleCallFunc() {
  uint64_t handle;
  CHECK_EQ(sock_.RecvAll(&handle, sizeof(handle)), sizeof(handle));
  PackedFunc* pf = reinterpret_cast<PackedFunc*>(handle);
  CallHandler([pf](TVMArgs args, TVMRetValue *rv) {
      pf->CallPacked(args, rv);
    });
}

void RPCSession::HandleException() {
  RPCArgBuffer ret;
  this->RecvPackedSeq(&ret);
  CHECK_EQ(ret.value.size(), 1U);
  CHECK_EQ(ret.tcode[0], kStr);
  std::ostringstream os;
  os << "Except caught from RPC call: " << ret.value[0].v_str;
  throw dmlc::Error(os.str());
}

void RPCSession::HandleCopyToRemote() {
  uint64_t handle, offset, size;
  TVMContext ctx;
  CHECK_EQ(sock_.RecvAll(&handle, sizeof(handle)), sizeof(handle));
  CHECK_EQ(sock_.RecvAll(&offset, sizeof(offset)), sizeof(offset));
  CHECK_EQ(sock_.RecvAll(&size, sizeof(size)), sizeof(size));
  CHECK_EQ(sock_.RecvAll(&ctx, sizeof(ctx)), sizeof(ctx));
  int succ = 0;
  if (ctx.device_type == kCPU) {
    CHECK_EQ(sock_.RecvAll(reinterpret_cast<char*>(handle) + offset, size),
             static_cast<size_t>(size));
  } else {
    temp_data_.resize(size+1);
    CHECK_EQ(sock_.RecvAll(&temp_data_[0], size),
             static_cast<size_t>(size));
    try {
      TVMContext cpu_ctx;
      cpu_ctx.device_type = kCPU;
      cpu_ctx.device_id = 0;
      DeviceAPI::Get(ctx)->CopyDataFromTo(
          temp_data_.data(), 0,
          reinterpret_cast<void*>(handle), offset,
          size, cpu_ctx, ctx, nullptr);
    } catch (const std::runtime_error &e) {
      TVMAPISetLastError(e.what());
      succ = -1;
    }
  }
  TVMValue ret_value;
  ret_value.v_handle = nullptr;
  int ret_tcode = kNull;
  SendReturnValue(succ, ret_value, ret_tcode);
}

void RPCSession::HandleCopyFromRemote() {
  uint64_t handle, offset, size;
  TVMContext ctx;
  CHECK_EQ(sock_.RecvAll(&handle, sizeof(handle)), sizeof(handle));
  CHECK_EQ(sock_.RecvAll(&offset, sizeof(offset)), sizeof(offset));
  CHECK_EQ(sock_.RecvAll(&size, sizeof(size)), sizeof(size));
  CHECK_EQ(sock_.RecvAll(&ctx, sizeof(ctx)), sizeof(ctx));
  if (ctx.device_type == kCPU) {
    RPCCode code = RPCCode::kCopyAck;
    CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
    CHECK_EQ(sock_.SendAll(reinterpret_cast<char*>(handle) + offset, size),
             static_cast<size_t>(size));
  } else {
    temp_data_.resize(size + 1);
    try {
      TVMContext cpu_ctx;
      cpu_ctx.device_type = kCPU;
      cpu_ctx.device_id = 0;
      DeviceAPI::Get(ctx)->CopyDataFromTo(
          reinterpret_cast<void*>(handle), offset,
          dmlc::BeginPtr(temp_data_), 0,
          size, ctx, cpu_ctx, nullptr);
      RPCCode code = RPCCode::kCopyAck;
      CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
      CHECK_EQ(sock_.SendAll(&temp_data_[0], size),
               static_cast<size_t>(size));
    } catch (const std::runtime_error &e) {
      RPCCode code = RPCCode::kException;
      CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
      TVMValue ret_value;
      ret_value.v_str = e.what();
      int ret_tcode = kStr;
      SendPackedSeq(&ret_value, &ret_tcode, 1);
    }
  }
}

void RPCSession::HandleReturn(TVMRetValue* rv) {
  RPCArgBuffer ret;
  this->RecvPackedSeq(&ret);
  CHECK_EQ(ret.value.size(), 1U);
  TVMArgValue argv = ret.AsTVMArgs()[0];
  *rv = argv;
}

TVMContext RPCSession::StripSessMask(TVMContext ctx) {
  int dev_type = ctx.device_type;
  CHECK_EQ(dev_type / kRPCSessMask, table_index_ + 1)
      << "Can only TVMContext related to the same remote sesstion";
  ctx.device_type = static_cast<DLDeviceType>(dev_type % kRPCSessMask);
  return ctx;
}

// packed Send sequence to the channel
void RPCSession::SendPackedSeq(
    const TVMValue* arg_values, const int* type_codes, int n) {
  CHECK_EQ(sock_.SendAll(&n, sizeof(n)), sizeof(n));
  CHECK_EQ(sock_.SendAll(type_codes, sizeof(int) * n), sizeof(int) * n);
  // Argument packing.
  for (int i = 0; i < n; ++i) {
    int tcode = type_codes[i];
    TVMValue value = arg_values[i];
    switch (tcode) {
      case kInt:
      case kUInt:
      case kFloat:
      case kTVMType: {
        CHECK_EQ(sock_.SendAll(&value, sizeof(TVMValue)), sizeof(TVMValue));
        break;
      }
      case kTVMContext: {
        value.v_ctx = StripSessMask(value.v_ctx);
        CHECK_EQ(sock_.SendAll(&value, sizeof(TVMValue)), sizeof(TVMValue));
        break;
      }
      case kHandle: {
        // always send handle in 64 bit.
        uint64_t handle = reinterpret_cast<uint64_t>(value.v_handle);
        CHECK_EQ(sock_.SendAll(&handle, sizeof(uint64_t)), sizeof(uint64_t));
        break;
      }
      case kArrayHandle: {
        DLTensor* arr = static_cast<DLTensor*>(value.v_handle);
        TVMContext ctx = StripSessMask(arr->ctx);
        uint64_t data = reinterpret_cast<uint64_t>(
            static_cast<RemoteSpace*>(arr->data)->data);
        CHECK_EQ(sock_.SendAll(&data, sizeof(uint64_t)), sizeof(uint64_t));
        CHECK_EQ(sock_.SendAll(&ctx, sizeof(ctx)), sizeof(ctx));
        CHECK_EQ(sock_.SendAll(&(arr->ndim), sizeof(int)), sizeof(int));
        CHECK_EQ(sock_.SendAll(&(arr->dtype), sizeof(DLDataType)), sizeof(DLDataType));
        CHECK_EQ(sock_.SendAll(arr->shape, sizeof(int64_t) * arr->ndim),
                 sizeof(int64_t) * arr->ndim);
        CHECK(arr->strides == nullptr)
            << "Donot support strided remote array";
        CHECK_EQ(arr->byte_offset, 0)
            << "Donot support send byte offset";
        break;
      }
      case kNull: break;
      case kStr: {
        const char* s = value.v_str;
        uint64_t len = strlen(s);
        CHECK_EQ(sock_.SendAll(&len, sizeof(len)), sizeof(len));
        CHECK_EQ(sock_.SendAll(s, sizeof(char) * len), sizeof(char) * len);
        break;
      }
      case kBytes: {
        TVMByteArray* bytes = static_cast<TVMByteArray*>(arg_values[i].v_handle);
        uint64_t len = bytes->size;
        CHECK_EQ(sock_.SendAll(&len, sizeof(len)), sizeof(len));
        CHECK_EQ(sock_.SendAll(bytes->data, sizeof(char) * len), sizeof(char) * len);
        break;
      }
      default: {
        LOG(FATAL) << "RPC cannot handle type " << TypeCode2Str(tcode);
        break;
      }
    }
  }
}

// Receive packed sequence from the channel
void RPCSession::RecvPackedSeq(RPCArgBuffer *buf) {
  int n;
  CHECK_EQ(sock_.RecvAll(&n, sizeof(n)), sizeof(n));
  buf->value.resize(n);
  buf->tcode.resize(n);
  buf->temp_bytes.clear();
  if (n != 0) {
    buf->tcode.resize(n);
    CHECK_EQ(sock_.RecvAll(buf->tcode.data(), sizeof(int) * n),
             sizeof(int) * n);
  }
  buf->value.resize(n);
  for (int i = 0; i < n; ++i) {
    int tcode = buf->tcode[i];
    TVMValue& value = buf->value[i];
    switch (tcode) {
      case kInt:
      case kUInt:
      case kFloat:
      case kTVMType:
      case kTVMContext: {
        CHECK_EQ(sock_.RecvAll(&value, sizeof(TVMValue)), sizeof(TVMValue));
        break;
      }
      case kHandle: {
          // always send handle in 64 bit.
        uint64_t handle;
        CHECK_EQ(sock_.RecvAll(&handle, sizeof(uint64_t)), sizeof(uint64_t));
        value.v_handle = reinterpret_cast<void*>(handle);
        break;
      }
      case kNull: {
        value.v_handle = nullptr;
        break;
      }
      case kStr:
      case kBytes: {
        uint64_t len;
        CHECK_EQ(sock_.RecvAll(&len, sizeof(len)), sizeof(len));
        std::unique_ptr<RPCByteArrayBuffer> temp(new RPCByteArrayBuffer());
        temp->data.resize(len);
        if (len != 0) {
          CHECK_EQ(sock_.RecvAll(&(temp->data[0]), sizeof(char) * len),
                   sizeof(char) * len);
        }
        if (tcode == kStr) {
          value.v_str = temp->data.c_str();
        } else {
          temp->arr.size = static_cast<size_t>(len);
          temp->arr.data = dmlc::BeginPtr(temp->data);
          value.v_handle = &(temp->arr);
        }
        buf->temp_bytes.emplace_back(std::move(temp));
        break;
      }
      case kArrayHandle: {
        std::unique_ptr<RPCDataArrayBuffer> temp(new RPCDataArrayBuffer());
        uint64_t handle;
        CHECK_EQ(sock_.RecvAll(&handle, sizeof(handle)), sizeof(handle));
        DLTensor& tensor = temp->tensor;
        tensor.data = reinterpret_cast<void*>(handle);
        CHECK_EQ(sock_.RecvAll(&(tensor.ctx), sizeof(TVMContext)), sizeof(TVMContext));
        CHECK_EQ(sock_.RecvAll(&(tensor.ndim), sizeof(int)), sizeof(int));
        CHECK_EQ(sock_.RecvAll(&(tensor.dtype), sizeof(DLDataType)), sizeof(DLDataType));
        temp->shape.resize(tensor.ndim);
        tensor.shape = temp->shape.data();
        CHECK_EQ(sock_.RecvAll(tensor.shape, tensor.ndim * sizeof(int64_t)),
                 tensor.ndim * sizeof(int64_t));
        tensor.strides = nullptr;
        tensor.byte_offset = 0;
        value.v_handle = &tensor;
        buf->temp_array.emplace_back(std::move(temp));
        break;
      }
      default: {
        LOG(FATAL) << "RPC cannot handle type " << TypeCode2Str(tcode);
        break;
      }
    }
  }
}

// Event handler functions
void RPCGetGlobalFunc(TVMArgs args, TVMRetValue* rv) {
  std::string name = args[0];
  auto *fp = tvm::runtime::Registry::Get(name);
  if (fp != nullptr) {
    *rv = static_cast<void*>(new tvm::runtime::PackedFunc(*fp));
  } else {
    *rv = nullptr;
  }
}

void RPCFreeFunc(TVMArgs args, TVMRetValue *rv) {
  void* handle = args[0];
  delete static_cast<PackedFunc*>(handle);
}

void RPCDevSetDevice(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  DeviceAPI::Get(ctx)->SetDevice(ctx);
}

void RPCDevGetAttr(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[1].operator int());
  if (kind == kExist) {
    DeviceAPI* api = DeviceAPI::Get(ctx, true);
    if (api != nullptr) {
      api->GetAttr(ctx, kind, rv);
    } else {
      *rv = 0;
    }
  } else {
    DeviceAPI::Get(ctx)->GetAttr(
        ctx, static_cast<DeviceAttrKind>(kind), rv);
  }
}

void RPCDevAllocData(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  uint64_t size = args[1];
  uint64_t alignment = args[2];
  void* data = DeviceAPI::Get(ctx)->AllocDataSpace(ctx, size, alignment);
  *rv = data;
}

void RPCDevFreeData(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  void* ptr = args[1];
  DeviceAPI::Get(ctx)->FreeDataSpace(ctx, ptr);
}

void RPCDevStreamSync(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  TVMStreamHandle handle = args[1];
  DeviceAPI::Get(ctx)->StreamSync(ctx, handle);
}

void RPCCopyAmongRemote(TVMArgs args, TVMRetValue *rv) {
  void* from = args[0];
  uint64_t from_offset = args[1];
  void* to = args[2];
  uint64_t to_offset = args[3];
  uint64_t size = args[4];
  TVMContext ctx_from = args[5];
  TVMContext ctx_to = args[6];
  TVMStreamHandle stream = args[7];
  TVMContext ctx = ctx_from;
  if (ctx.device_type == kCPU) {
    ctx = ctx_to;
  } else {
    CHECK(ctx_to.device_type == kCPU ||
          ctx_to.device_type == ctx_from.device_type)
        << "Can not copy across different ctx types directly";
  }
  DeviceAPI::Get(ctx)->CopyDataFromTo(
      from, from_offset,
      to, to_offset,
      size, ctx_from, ctx_to, stream);
}

void RPCModuleLoad(TVMArgs args, TVMRetValue *rv) {
  static const PackedFunc* fsys_load_ = nullptr;
  if (fsys_load_ == nullptr) {
    fsys_load_ = runtime::Registry::Get("tvm.contrib.rpc.server.load_module");
    CHECK(fsys_load_ != nullptr);
  }
  std::string file_name = args[0];
  TVMRetValue ret = (*fsys_load_)(file_name);
  Module m = ret;
  *rv = static_cast<void*>(new Module(m));
}

void RPCModuleFree(TVMArgs args, TVMRetValue *rv) {
  void* mhandle = args[0];
  delete static_cast<Module*>(mhandle);
}

void RPCModuleGetFunc(TVMArgs args, TVMRetValue *rv) {
  void* mhandle = args[0];
  PackedFunc pf = static_cast<Module*>(mhandle)->GetFunction(
      args[1], false);
  *rv = static_cast<void*>(new PackedFunc(pf));
}

void RPCModuleGetSource(TVMArgs args, TVMRetValue *rv) {
  void* mhandle = args[0];
  std::string fmt = args[1];
  *rv = (*static_cast<Module*>(mhandle))->GetSource(fmt);
}

RPCCode RPCSession::HandleNextEvent(TVMRetValue *rv) {
  RPCCode code;
  CHECK_EQ(sock_.RecvAll(&code, sizeof(int)), sizeof(int));
  switch (code) {
    case RPCCode::kCallFunc: HandleCallFunc(); break;
    case RPCCode::kReturn: HandleReturn(rv); break;
    case RPCCode::kException: HandleException(); break;
    case RPCCode::kCopyFromRemote: HandleCopyFromRemote(); break;
    case RPCCode::kCopyToRemote: HandleCopyToRemote(); break;
    case RPCCode::kShutdown: break;
    // system functions
    case RPCCode::kFreeFunc: CallHandler(RPCFreeFunc); break;
    case RPCCode::kGetGlobalFunc: CallHandler(RPCGetGlobalFunc); break;
    case RPCCode::kDevSetDevice: CallHandler(RPCDevSetDevice); break;
    case RPCCode::kDevGetAttr: CallHandler(RPCDevGetAttr); break;
    case RPCCode::kDevAllocData: CallHandler(RPCDevAllocData); break;
    case RPCCode::kDevFreeData: CallHandler(RPCDevFreeData); break;
    case RPCCode::kDevStreamSync: CallHandler(RPCDevStreamSync); break;
    case RPCCode::kCopyAmongRemote: CallHandler(RPCCopyAmongRemote); break;
    case RPCCode::kModuleLoad: CallHandler(RPCModuleLoad); break;
    case RPCCode::kModuleFree: CallHandler(RPCModuleFree); break;
    case RPCCode::kModuleGetFunc: CallHandler(RPCModuleGetFunc); break;
    case RPCCode::kModuleGetSource: CallHandler(RPCModuleGetSource); break;
    default: LOG(FATAL) << "Unknown event " << static_cast<int>(code);
  }
  return code;
}
}  // namespace runtime
}  // namespace tvm
