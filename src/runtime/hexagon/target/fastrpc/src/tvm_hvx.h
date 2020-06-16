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

#ifndef TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_SRC_TVM_HVX_H_
#define TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_SRC_TVM_HVX_H_

// Utility providing functions for accessing the Hexagon Vector Extensions
// (HVX) hardware.

#include <cstdint>

namespace hvx {

enum mode_t : uint32_t {
  MODE_DONT_CARE = 0, /*!< Don't-care, just use whatever current mode is. */
  MODE_64B,           /*!< 64 byte HVX vector width.                      */
  MODE_128B           /*!< 128 byte HVX vector width.                     */
};

/*!
 * \brief HVX configuration data.
 */
struct config_t {
  int num_reserved;  /*!< Number of reserved HVX units.                  */
  bool temp_reserve; /*!< Indicates that HVX pool reservation is         */
                     /*!< temporary and needs to be released after use.  */
  mode_t mode;       /*!< Configured HVX mode.                           */
  int vlen;          /*!< Configured HVX vector width (64 or 128 bytes). */
  int num_threads;   /*!< Number of threads that can lock HVX units.     */
};

/*!
 * \brief
 *   This function reserves HVX units for the protection domain to which
 *   the caller belongs. Reservation is optional before locking HVX units.
 *   Typically it would be called by applications that want to guarantee
 *   up front that the requested number of HVX units will be available
 *   for the duration of the application.
 *
 * \param num_units
 *   Number of HVX units to reserve. 0 indicates to reserve all the units
 *   present in the given target. > 0 indicates the number of single HVX
 *   units to reserve. Mode (64 byte vs. 128 byte) is not specified.
 *
 * \return
 *   The number of HVX units (in terms of 64 byte single units) successfully
 *   reserved. The return value of -1 indicates no HVX hardware is available
 *   on the target.
 */
int reserve(unsigned num_units);

/*!
 * \brief
 *   This function releases all HVX unit from reservation. A call to this
 *   function nullifies all previous calls to reserve HVX units from within
 *   this worker pool's protection domain.
 *
 * \return
 *   0 on success, -1 if there was an error.
 */
int unreserve();

/*!
 * \brief
 *   This function turns on the HVX hardware. It must be called sometime
 *   before (possibly multiple) software threads lock HVX units.
 *
 * \return
 *   0 on success, -1 if there was an error.
 */
int power_on();

/*!
 * \brief
 *   This function turns off the HVX hardware. It must be called sometime
 *   after all threads have unlocked their HVX units.
 *
 * \return
 *   0 on success, -1 if there was an error.
 */
int power_off();

/*!
 * \brief
 *   This function locks the HVX units for the calling threads.
 *
 * \param mode
 *   The HVX mode.
 *
 * \return
 *   0 on success, -1 if there was an error.
 */
int lock(mode_t mode);

/*!
 * \brief
 *   This function unlocks the HVX units for the calling threads.
 *
 * \return
 *   0 on success, -1 if there was an error.
 */
int unlock();

/*!
 * \brief
 *   This function performs preparations for multithreaded job.
 *   It does so by filling out data members in the configuration
 *   structure passed as a parameter, and by setting up the hardware:
 *   - it performs a temporary reservation of HVX units, if no units
 *     have yet been reserved,
 *   - it powers on the HVX hardware.
 *
 * \param hvx_config
 *   Structure describing the HVX configuration. Two data members
 *   must be set prior to calling \ref prepare_mt_job:
 *   \ref num_reserved, indicating the number of previously reserved
 *   HVX units (can be 0), and \ref mode indicating the HVX mode.
 *
 * \return
 *   0 on success, -1 if there was an error.
 */
int prepare_mt_job(config_t* hvx_config);

/*!
 * \brief
 *   This function cleans up after \ref prepare_mt_job, in particular
 *   it releases temporarily reserved HVX units and turns the HVX
 *   hardware off.
 *
 * \return
 *   0 on success, -1 if there was an error.
 */
int cleanup_mt_job(const config_t* hvx_config);

}  // namespace hvx

#endif  // TVM_RUNTIME_HEXAGON_TARGET_FASTRPC_SRC_TVM_HVX_H_
