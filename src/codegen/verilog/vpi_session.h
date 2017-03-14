/*!
 *  Copyright (c) 2017 by Contributors
 * \file vpi_session.h
 * \brief IPC session call to verilog simulator via VPI.
 */
#ifndef TVM_CODEGEN_VERILOG_VPI_SESSION_H_
#define TVM_CODEGEN_VERILOG_VPI_SESSION_H_

#include <tvm/base.h>
#include <vector>
#include <string>
#include "../../common/pipe.h"
#include "../../../verilog/tvm_vpi.h"

namespace tvm {
namespace codegen {
// node containers
class VPISessionNode;
class VPIHandleNode;
class VPIHandle;

/*! \brief Environment */
class VPISession : public NodeRef {
 public:
  VPISession() {}
  explicit VPISession(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief Get handle by name.
   * \param name The name of the handle.
   */
  VPIHandle operator[](const std::string& name) const;
  /*!
   * \brief Yield control back to the simulator
   *  Block until next cycle.
   */
  void yield();
  /*!
   * \brief Shutdown the session.
   */
  void shutdown();
  /*!
   * \brief Create new session by giving a read and write pipe to VPI process.
   * \param h_pipe_read a read pipe from VPI process.
   * \param h_pipe_write a write pipe from VPI process.
   */
  static VPISession make(int h_pipe_read, int h_pipe_write);
  // Internal methods.
  using ContainerType = VPISessionNode;

 private:
  friend class VPIHandle;
  inline VPISessionNode* get() const;
  // Get handle by name
  VPIHandle GetByName(const std::string& name, vpi::VPIRawHandle handle) const;
};

/*! \brief VPI Handle */
class VPIHandle : public NodeRef {
 public:
  VPIHandle() {}
  explicit VPIHandle(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief Get handle by name.
   * \param name The name of the handle.
   */
  VPIHandle operator[](const std::string& name) const;
  /*! \return number of bits */
  int size() const;
  /*!
   * \brief Set int value to the handle.
   * \param value The value to set.
   */
  void put_int(int value);
  /*!
   * \brief Get int value from handle.
   * \return The result int value.
   */
  int get_int() const;
  /*! \return Name of the handle. */
  std::string name() const;
  /*!
   * \brief Put byte vector into the handle.
   * \param vec The vector to be put.
   * \return The result int value.
   */
  void put_vec(const std::vector<vpi::VPIVecVal>& vec) const;
  /*!
   * \brief Get byte vector from handle.
   * \param vec The result data container.
   */
  void get_vec(std::vector<vpi::VPIVecVal>* vec) const;
  // Internal methods
  using ContainerType = VPIHandleNode;

 private:
  inline VPIHandleNode* get() const;
};
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_VERILOG_VPI_SESSION_H_
