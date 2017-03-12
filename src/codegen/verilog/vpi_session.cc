/*!
 *  Copyright (c) 2017 by Contributors
 * \file vpi_session.cc
 * \brief IPC session call to verilog simulator via VPI.
 */
#include <tvm/api_registry.h>
#include "./vpi_session.h"

namespace tvm {
namespace codegen {

using namespace vpi;

/*! \brief Container for session. */
class VPISessionNode : public Node {
 public:
  // Whether in control.
  bool in_control{false};
  // Internal reader and writer.
  common::Pipe reader;
  common::Pipe writer;

  // internal constructor
  VPISessionNode(int h_pipe_read, int h_pipe_write)
      : reader(h_pipe_read), writer(h_pipe_write) {
  }
  ~VPISessionNode() {
    if (in_control) {
      VPIReturnCode cd;
      writer.Write(kShutDown);
      reader.Read(&cd);
    }
    reader.Close();
    writer.Close();
  }
  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
  }
  void ReadExpect(VPIReturnCode rcode) {
    VPIReturnCode code;
    CHECK(reader.Read(&code));
    CHECK_EQ(code, rcode) << "Error in simulation";
  }

  static constexpr const char* _type_key = "VPISession";
  TVM_DECLARE_NODE_TYPE_INFO(VPISessionNode, Node);
};

/*! \brief Container for handle */
class VPIHandleNode : public Node {
 public:
  // The internal session.
  VPISession sess;
  // Internal handle
  VPIRawHandle handle;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("sess", &sess);
  }
  static VPIHandle make(const VPISession& sess, VPIRawHandle handle) {
    std::shared_ptr<VPIHandleNode> n =
        std::make_shared<VPIHandleNode>();
    n->sess = sess;
    n->handle = handle;
    return VPIHandle(n);
  }

  static constexpr const char* _type_key = "VPIHandle";
  TVM_DECLARE_NODE_TYPE_INFO(VPIHandleNode, Node);
};

// Inline implementations
inline VPISessionNode* VPISession::get() const {
  return static_cast<VPISessionNode*>(node_.get());
}
inline VPIHandleNode* VPIHandle::get() const {
  return static_cast<VPIHandleNode*>(node_.get());
}

VPISession VPISession::make(int h_pipe_read, int h_pipe_write) {
  std::shared_ptr<VPISessionNode> n = std::make_shared<VPISessionNode>(
      h_pipe_read, h_pipe_write);
  n->ReadExpect(kPosEdgeTrigger);
  n->in_control = true;
  return VPISession(n);
}

VPIHandle VPISession::operator[](const std::string& name) const {
  return GetByName(name, nullptr);
}

VPIHandle VPISession::GetByName(const std::string& name, VPIRawHandle handle) const {
  VPISessionNode* n = get();
  CHECK(n->in_control);
  n->writer.Write(kGetHandleByName);
  n->writer.Write(name);
  n->writer.Write(handle);
  n->ReadExpect(kSuccess);
  CHECK(n->reader.Read(&handle));
  CHECK(handle != nullptr)
      << "Cannot find handle with name=" << name;
  return VPIHandleNode::make(*this, handle);
}

void VPISession::yield() {
  VPISessionNode* n = get();
  CHECK(n->in_control);
  n->writer.Write(kYield);
  n->ReadExpect(kSuccess);
  n->in_control = false;
  n->ReadExpect(kPosEdgeTrigger);
  n->in_control = true;
}

void VPISession::shutdown() {
  VPISessionNode* n = get();
  if (n->in_control) {
    n->writer.Write(kShutDown);
    n->ReadExpect(kSuccess);
    n->in_control = false;
  }
}

int VPIHandle::size() const {
  VPIHandleNode* h = get();
  VPISessionNode* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetSize);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  int value;
  CHECK(n->reader.Read(&value));
  return value;
}

void VPIHandle::put_int(int value) {
  VPIHandleNode* h = get();
  VPISessionNode* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kPutInt32);
  n->writer.Write(h->handle);
  n->writer.Write(value);
  n->ReadExpect(kSuccess);
}

int VPIHandle::get_int() const {
  VPIHandleNode* h = get();
  VPISessionNode* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetInt32);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  int value;
  CHECK(n->reader.Read(&value));
  return value;
}

std::string VPIHandle::name() const {
  VPIHandleNode* h = get();
  VPISessionNode* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kGetName);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  std::string str;
  CHECK(n->reader.Read(&str));
  return str;
}

void VPIHandle::put_vec(const std::vector<VPIVecVal>& vec) const {
  VPIHandleNode* h = get();
  VPISessionNode* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kPutVec);
  n->writer.Write(h->handle);
  n->writer.Write(vec);
  n->ReadExpect(kSuccess);
}

void VPIHandle::get_vec(std::vector<VPIVecVal>* vec) const {
  VPIHandleNode* h = get();
  VPISessionNode* n = h->sess.get();
  CHECK(n->in_control);
  n->writer.Write(kPutVec);
  n->writer.Write(h->handle);
  n->ReadExpect(kSuccess);
  CHECK(n->reader.Read(&vec));
}

VPIHandle VPIHandle::operator[](const std::string& name) const {
  VPIHandleNode* h = get();
  return h->sess.GetByName(name, h->handle);
}

// API registration
TVM_REGISTER_API(_vpi_SessMake)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = VPISession::make(args[0], args[1]);
  });

TVM_REGISTER_API(_vpi_SessGetHandleByName)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPISession().operator[](args[1]);
  });

TVM_REGISTER_API(_vpi_SessYield)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator VPISession().yield();
  });

TVM_REGISTER_API(_vpi_SessShutdown)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator VPISession().shutdown();
  });

TVM_REGISTER_API(_vpi_HandlePutInt)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator VPIHandle().put_int(args[1]);
  });

TVM_REGISTER_API(_vpi_HandleGetInt)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().get_int();
  });

TVM_REGISTER_API(_vpi_HandleGetName)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().name();
  });

TVM_REGISTER_API(_vpi_HandleGetSize)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().size();
  });

TVM_REGISTER_API(_vpi_HandleGetHandleByName)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator VPIHandle().operator[](args[1]);
  });

}  // namespace codegen
}  // namespace tvm
