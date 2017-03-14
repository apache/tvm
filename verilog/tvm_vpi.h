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
  kGetName,
  kGetInt32,
  kPutInt32,
  kGetSize,
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
