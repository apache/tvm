/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm_vpi.h
 * \brief Messages passed around VPI used for simulation.
 */
#ifndef VERILOG_TVM_VPI_H_
#define VERILOG_TVM_VPI_H_

namespace tvm {
namespace vpi {

enum VPICallCode : int {
  kGetHandleByName,
  kGetHandleByIndex,
  kGetStrProp,
  kGetIntProp,
  kGetInt32,
  kPutInt32,
  kGetVec,
  kPutVec,
  kYield,
  kShutDown
};

enum VPIReturnCode : int {
  kPosEdgeTrigger = 0,
  kSuccess = 1,
  kFail = 2
};

// VPI type code as in IEEE standard.
enum VPITypeCode {
  kVPIModule = 32
};

// VPI property code as in IEEE standard.
enum VPIPropCode {
  kVPIType = 1,
  kVPIFullName = 3,
  kVPISize = 4,
  kVPIDefName = 9
};

/*! \brief The vector value used in trasmission */
struct VPIVecVal {
  int aval;
  int bval;
};

/*! \brief User facing vpi handle. */
typedef void* VPIRawHandle;

}  // namespace vpi
}  // namespace tvm
#endif  // VERILOG_TVM_VPI_H_
