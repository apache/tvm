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

package vta.shell

import chisel3._
import chisel3.util._
import vta.util.config._
import vta.interface.axi._

/** PynqConfig. Shell configuration for Pynq */
class PynqConfig extends Config((site, here, up) => {
  case ShellKey => ShellParams(
    hostParams = AXIParams(
      addrBits = 16,
      dataBits = 32),
    memParams = AXIParams(
      addrBits = 32,
      dataBits = 64),
    vcrParams = VCRParams(),
    vmeParams = VMEParams())
})

/** F1Config. Shell configuration for F1 */
class F1Config extends Config((site, here, up) => {
  case ShellKey => ShellParams(
    hostParams = AXIParams(
      addrBits = 16,
      dataBits = 32),
    memParams = AXIParams(
      addrBits = 64,
      dataBits = 64),
    vcrParams = VCRParams(),
    vmeParams = VMEParams())
})
