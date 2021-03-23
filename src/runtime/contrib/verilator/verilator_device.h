/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/verilator/verilator_device.h
 * \brief Use external verilator device.
 */

#ifndef TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_DEVICE_H_
#define TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_DEVICE_H_

#include <tvm/runtime/c_runtime_api.h>

namespace tvm {
namespace runtime {
namespace contrib {

/*! \brief Verilator device resource context  */
typedef void* VerilatorHandle;

/*!
 * \brief Allocate a verilator device resource handle
 * \return The verilator device handle.
 */
extern "C" TVM_DLL VerilatorHandle VerilatorAlloc();

/*!
 * \brief Free a verilator device handle
 * \param handle The verilator device handle to be freed.
 */
extern "C" TVM_DLL void VerilatorDealloc(VerilatorHandle handle);

/*!
 * \brief Read verilator register or memory
 * \param handle The verilator device handle.
 * \param id The register or memory identifier.
 * \param addr The register or memory address (word-level).
 * \return The value of register or memory.
 */
extern "C" TVM_DLL int VerilatorRead(VerilatorHandle handle, int id, int addr);

/*!
 * \brief Write verilator register or memory
 * \param handle The verilator device handle.
 * \param id The register or memory identifier.
 * \param addr The register or memory address (word-level).
 * \param value The value of register or memory.
 */
extern "C" TVM_DLL void VerilatorWrite(VerilatorHandle handle, int id, int addr, int value);

/*!
 * \brief Reset Verilator for n clock cycles
 * \param handle The verilator device handle.
 * \param n The number of reset cycles.
 */
extern "C" TVM_DLL void VerilatorReset(VerilatorHandle handle, int n);

/*!
 * \brief Run Verilator for n clock cycles
 * \param handle The verilator device handle.
 * \param n The number of run cycles.
 */
extern "C" TVM_DLL void VerilatorRun(VerilatorHandle handle, int n);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_DEVICE_H_
