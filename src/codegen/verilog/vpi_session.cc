/*!
 *  Copyright (c) 2017 by Contributors
 * \file vpi_session.cc
 * \brief IPC session call to verilog simulator via VPI.
 */
#include <tvm/api_registry.h>
#include "vpi_session.h"

namespace tvm {
namespace codegen {

using namespace vpi;

// helper class to get the node.
class VPISessionEntry {
 public:
  // Whether in control.
  bool in_control{false};
  // Internal reader and writer.
  common::Pipe reader;
  common::Pipe writer;
  // internal constructor
  VPISessionEntry(int h_pipe_read, int h_pipe_write)
      : reader(h_pipe_read), writer(h_pipe_write) {
  }
  ~VPISessionEntry() {
    if (in_control) {
      VPIReturnCode cd;
      writer.Write(kShutDown);
      reader.Read(&cd);
    }
    reader.Close();
    writer.Close();
  }
  void ReadExpect(VPIReturnCode rcode) {
    VPIReturnCode code;
    CHECK(reader.Read(&code));
    CHECK_EQ(code, rcode) << "Error in simulation";
  }
};

// Inline implementations
inline VPISessionNode* VPISession::get() const {
  return static_cast<VPISessionNode*>(node_.get());
}
inline VPIHandleNode* VPIHandle::get() const {
  return static_cast<VPIHandleNode*>(node_.get());
}

VPIHandle VPIHandleCreate(
    const std::shared_ptr<VPISessionEntry>& sess,
    VPIRawHandle handle) {
  auto n = make_node<VPIHandleNode>();
  n->sess = sess;
  n->handle = handle;
  return VPIHandle(n);
}

VPIHandle GetHandleByName(
    const std::shared_ptr<VPISessionEntry>& sess,
    const std::string& name,
    VPIRawHandle handle,
    bool allow_undefined) {
  VPISessionEntry* n = sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetHandleByName);
  n->writer.Write(name);
  n->writer.Write(handle);
  n->ReadExpect(kSuccess);
  CHECK(n->reader.Read(&handle));
  if (handle != nullptr) {
    return VPIHandleCreate(sess, handle);
  } else {
    CHECK(allow_undefined)
        << "Cannot find handle with name=" << name;
    return VPIHandle();
  }
}

std::string VPIGetStrProp(VPIHandleNode* h, int code) {
  VPISessionEntry* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetStrProp);
  n->writer.Write(code);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  std::string str;
  CHECK(n->reader.Read(&str));
  return str;
}

int VPIGetIntProp(VPIHandleNode* h, int code) {
  VPISessionEntry* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetIntProp);
  n->writer.Write(code);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  int value;
  CHECK(n->reader.Read(&value));
  return value;
}

VPISession VPISession::make(int h_pipe_read, int h_pipe_write) {
  auto n = make_node<VPISessionNode>();
  n->sess = std::make_shared<VPISessionEntry>(h_pipe_read, h_pipe_write);
  n->sess->in_control = true;
  VPISession sess(n);
  // The custom module handles
  std::vector<VPIRawHandle> mod_handles;
  n->sess->reader.Read(&mod_handles);
  n->sess->ReadExpect(kPosEdgeTrigger);
  // start Initialize the callbacks
  for (VPIRawHandle raw_h : mod_handles) {
    VPIHandle h = VPIHandleCreate(n->sess, raw_h);
    CHECK_EQ(VPIGetIntProp(h.get(), kVPIType), kVPIModule)
        << "Expect pass modules to $tvm_session after clk";
    std::string def = VPIGetStrProp(h.get(), kVPIDefName);
    std::string callback_name = "_vpi_module_" + def;
    const PackedFunc* f = runtime::Registry::Get(callback_name);
    CHECK(f != nullptr)
        << "Cannot find definition for tvm vpi module " << def;
    PackedFunc cb = (*f)(h);
    n->posedge_end_callbacks.push_back(cb);
  }
  return sess;
}

VPIHandle VPISession::operator[](const std::string& name) const {
  return GetHandleByName(get()->sess, name, nullptr, false);
}
VPIHandle VPISession::GetByName(const std::string& name,
                                bool allow_undefined) const {
  return GetHandleByName(get()->sess, name, nullptr, true);
}

void VPISession::yield() {
  VPISessionEntry* n = get()->sess.get();
  CHECK(n->in_control);
  for (const PackedFunc& f : get()->posedge_end_callbacks) {
    f();
  }
  n->writer.Write(kYield);
  n->ReadExpect(kSuccess);
  n->in_control = false;
  n->ReadExpect(kPosEdgeTrigger);
  n->in_control = true;
}

void VPISession::shutdown() {
  VPISessionEntry* n = get()->sess.get();
  if (n->in_control) {
    n->writer.Write(kShutDown);
    n->ReadExpect(kSuccess);
    n->in_control = false;
  }
}

int VPIHandle::size() const {
  return VPIGetIntProp(get(), kVPISize);
}

void VPIHandle::put_int(int value) {
  VPIHandleNode* h = get();
  VPISessionEntry* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kPutInt32);
  n->writer.Write(h->handle);
  n->writer.Write(value);
  n->ReadExpect(kSuccess);
}

int VPIHandle::get_int() const {
  VPIHandleNode* h = get();
  VPISessionEntry* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetInt32);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  int value;
  CHECK(n->reader.Read(&value));
  return value;
}

std::string VPIHandle::name() const {
  return VPIGetStrProp(get(), kVPIFullName);
}

void VPIHandle::put_vec(const std::vector<VPIVecVal>& vec) const {
  VPIHandleNode* h = get();
  VPISessionEntry* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kPutVec);
  n->writer.Write(h->handle);
  n->writer.Write(vec);
  n->ReadExpect(kSuccess);
}

void VPIHandle::get_vec(std::vector<VPIVecVal>* vec) const {
  VPIHandleNode* h = get();
  VPISessionEntry* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetVec);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  CHECK(n->reader.Read(vec));
}

VPIHandle VPIHandle::operator[](const std::string& name) const {
  VPIHandleNode* h = get();
  return GetHandleByName(h->sess, name, h->handle, false);
}

// API registration
TVM_REGISTER_API("_vpi_SessMake")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = VPISession::make(args[0], args[1]);
  });

TVM_REGISTER_API("_vpi_SessGetHandleByName")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPISession().operator[](args[1]);
  });

TVM_REGISTER_API("_vpi_SessYield")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator VPISession().yield();
  });

TVM_REGISTER_API("_vpi_SessShutdown")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator VPISession().shutdown();
  });

TVM_REGISTER_API("_vpi_HandlePutInt")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator VPIHandle().put_int(args[1]);
  });

TVM_REGISTER_API("_vpi_HandleGetInt")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().get_int();
  });

TVM_REGISTER_API("_vpi_HandleGetName")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().name();
  });

TVM_REGISTER_API("_vpi_HandleGetSize")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().size();
  });

TVM_REGISTER_API("_vpi_HandleGetHandleByName")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().operator[](args[1]);
  });

}  // namespace codegen
}  // namespace tvm
