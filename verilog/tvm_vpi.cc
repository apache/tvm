/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm_vpi.cc
 * \brief Messages passed around VPI used for simulation.
 */
#include <dmlc/logging.h>
#include <vpi_user.h>
#include <cstdlib>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include "tvm_vpi.h"
#include "../src/common/pipe.h"

namespace tvm {
namespace vpi {
// standard consistency checks
static_assert(sizeof(vpiHandle) == sizeof(VPIRawHandle),
              "VPI standard");
// type codes
static_assert(vpiModule == kVPIModule, "VPI standard");
// Property code
static_assert(vpiType == kVPIType, "VPI standard");
static_assert(vpiFullName == kVPIFullName, "VPI standard");
static_assert(vpiSize == kVPISize, "VPI standard");
static_assert(vpiDefName == kVPIDefName, "VPI standard");

// IPC client for VPI
class IPCClient {
 public:
  // constructor
  IPCClient(int64_t hread, int64_t hwrite)
      : reader_(hread), writer_(hwrite) {
  }
  void Init() {
    vpiHandle argv = vpi_handle(vpiSysTfCall, 0);
    vpiHandle arg_iter = vpi_iterate(vpiArgument, argv);
    clock_ = vpi_scan(arg_iter);
    std::vector<VPIRawHandle> handles;
    while (vpiHandle h = vpi_scan(arg_iter)) {
      handles.push_back(h);
    }
    writer_.Write(handles);
    PutInt(clock_, 0);
  }
  int Callback() {
    if (!GetInt(clock_)) {
      try {
        return AtNegEdge();
      } catch (const std::runtime_error& e) {
        reader_.Close();
        writer_.Close();
        vpi_printf("ERROR: encountered %s\n", e.what());
        vpi_control(vpiFinish, 1);
        return 0;
      }
    } else {
      return 0;
    }
  }
  // called at neg edge.
  int AtNegEdge() {
    // This is actually called at neg-edge
    // The put values won't take effect until next neg-edge.
    // This allow us to see the registers before snc
    writer_.Write(kPosEdgeTrigger);
    VPICallCode rcode;
    VPIRawHandle handle;
    int32_t index, value;

    while (true) {
      CHECK(reader_.Read(&rcode));
      switch (rcode) {
        case kGetHandleByName: {
          std::string str;
          CHECK(reader_.Read(&str));
          CHECK(reader_.Read(&handle));
          handle = vpi_handle_by_name(
              str.c_str(), static_cast<vpiHandle>(handle));
          writer_.Write(kSuccess);
          writer_.Write(handle);
          break;
        }
        case kGetHandleByIndex: {
          CHECK(reader_.Read(&handle));
          CHECK(reader_.Read(&index));
          handle = vpi_handle_by_index(
              static_cast<vpiHandle>(handle), index);
          writer_.Write(kSuccess);
          writer_.Write(handle);
          break;
        }
        case kGetStrProp: {
          CHECK(reader_.Read(&value));
          CHECK(reader_.Read(&handle));
          std::string prop = vpi_get_str(
              value, static_cast<vpiHandle>(handle));
          writer_.Write(kSuccess);
          writer_.Write(prop);
          break;
        }
        case kGetIntProp: {
          CHECK(reader_.Read(&value));
          CHECK(reader_.Read(&handle));
          value = vpi_get(value, static_cast<vpiHandle>(handle));
          writer_.Write(kSuccess);
          writer_.Write(value);
          break;
        }
        case kGetInt32: {
          CHECK(reader_.Read(&handle));
          value = GetInt(static_cast<vpiHandle>(handle));
          writer_.Write(kSuccess);
          writer_.Write(value);
          break;
        }
        case kPutInt32: {
          CHECK(reader_.Read(&handle));
          CHECK(reader_.Read(&value));
          CHECK(handle != clock_) << "Cannot write to clock";
          PutInt(static_cast<vpiHandle>(handle), value);
          writer_.Write(kSuccess);
          break;
        }
        case kGetVec: {
          CHECK(reader_.Read(&handle));
          vpiHandle h = static_cast<vpiHandle>(handle);
          int bits = vpi_get(vpiSize, h);
          int nwords = (bits + 31) / 32;
          s_vpi_value  value_s;
          value_s.format = vpiVectorVal;
          vpi_get_value(h, &value_s);
          vec_buf_.resize(nwords);
          for (size_t i = 0; i < vec_buf_.size(); ++i) {
            vec_buf_[i].aval = value_s.value.vector[i].aval;
            vec_buf_[i].bval = value_s.value.vector[i].bval;
          }
          writer_.Write(kSuccess);
          writer_.Write(vec_buf_);
          break;
        }
        case kPutVec: {
          CHECK(reader_.Read(&handle));
          CHECK(reader_.Read(&vec_buf_));
          CHECK(handle != clock_) << "Cannot write to clock";
          vpiHandle h = static_cast<vpiHandle>(handle);
          svec_buf_.resize(vec_buf_.size());
          for (size_t i = 0; i < vec_buf_.size(); ++i) {
            svec_buf_[i].aval = vec_buf_[i].aval;
            svec_buf_[i].bval = vec_buf_[i].bval;
          }
          s_vpi_value  value_s;
          s_vpi_time time_s;
          time_s.type = vpiSimTime;
          time_s.high = 0;
          time_s.low  = 10;
          value_s.format = vpiVectorVal;
          value_s.value.vector = &svec_buf_[0];
          vpi_put_value(h, &value_s, &time_s, vpiTransportDelay);
          writer_.Write(kSuccess);
          break;
        }
        case kYield: {
          writer_.Write(kSuccess);
          return 0;
        }
        case kShutDown : {
          writer_.Write(kSuccess);
          vpi_control(vpiFinish, 0);
          return 0;
        }
      }
    }
  }
  // Create a new FSM from ENV.
  static IPCClient* Create() {
    const char* d_read = getenv("TVM_DREAD_PIPE");
    const char* d_write = getenv("TVM_DWRITE_PIPE");
    const char* h_read = getenv("TVM_HREAD_PIPE");
    const char* h_write = getenv("TVM_HWRITE_PIPE");
    if (d_write == nullptr ||
        d_read == nullptr ||
        h_read == nullptr ||
        h_write == nullptr) {
      vpi_printf("ERROR: need environment var TVM_READ_PIPE, TVM_WRITE_PIPE\n");
      vpi_control(vpiFinish, 1);
      return nullptr;
    }
    // close host side pipe.
    common::Pipe(atoi(h_read)).Close();
    common::Pipe(atoi(h_write)).Close();
    IPCClient* client = new IPCClient(atoi(d_read), atoi(d_write));
    client->Init();
    return client;
  }
  // Get integer from handle.
  static int GetInt(vpiHandle h) {
    s_vpi_value  value_s;
    value_s.format = vpiIntVal;
    vpi_get_value(h, &value_s);
    return value_s.value.integer;
  }
  // Put integer into handle.
  static void PutInt(vpiHandle h, int value) {
    s_vpi_value  value_s;
    s_vpi_time time_s;
    time_s.type = vpiSimTime;
    time_s.high = 0;
    time_s.low  = 10;
    value_s.format = vpiIntVal;
    value_s.value.integer = value;
    vpi_put_value(h, &value_s, &time_s, vpiTransportDelay);
  }
  // Handles
  vpiHandle clock_;
  // the communicator
  common::Pipe reader_, writer_;
  // data buf
  std::vector<VPIVecVal> vec_buf_;
  std::vector<s_vpi_vecval> svec_buf_;
};
}  // namespace vpi
}  // namespace tvm

extern "C" {
static PLI_INT32 tvm_host_clock_cb(p_cb_data cb_data) {
  return reinterpret_cast<tvm::vpi::IPCClient*>(
      cb_data->user_data)->Callback();
}

static PLI_INT32 tvm_init(char* cb) {
  s_vpi_value  value_s;
  s_vpi_time  time_s;
  s_cb_data  cb_data_s;
  tvm::vpi::IPCClient* client = tvm::vpi::IPCClient::Create();
  if (client) {
    cb_data_s.user_data = reinterpret_cast<char*>(client);
    cb_data_s.reason = cbValueChange;
    cb_data_s.cb_rtn = tvm_host_clock_cb;
    cb_data_s.time = &time_s;
    cb_data_s.value = &value_s;
    time_s.type = vpiSuppressTime;
    value_s.format = vpiIntVal;
    cb_data_s.obj  = client->clock_;
    vpi_register_cb(&cb_data_s);
  } else {
    vpi_printf("ERROR: canot initalize host\n");
    vpi_control(vpiFinish, 1);
  }
  return 0;
}

void tvm_vpi_register() {
  s_vpi_systf_data tf_data;
  tf_data.type = vpiSysTask;
  tf_data.tfname = "$tvm_session";
  tf_data.calltf = tvm_init;
  tf_data.compiletf = nullptr;
  tf_data.sizetf = nullptr;
  tf_data.user_data = nullptr;
  vpi_register_systf(&tf_data);
}

void (*vlog_startup_routines[])() = {
  tvm_vpi_register,
  0
};
}  // extern "C"
