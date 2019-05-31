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

package vta.core

import vta.util.config._

/** CoreConfig.
  *
  * This is one supported configuration for VTA. This file will
  * be eventually filled out with class configurations that can be
  * mixed/matched with Shell configurations for different backends.
  */
class CoreConfig extends Config((site, here, up) => {
  case CoreKey => CoreParams(
    batch = 1,
    blockOut = 16,
    blockIn = 16,
    inpBits = 8,
    wgtBits = 8,
    uopBits = 32,
    accBits = 32,
    outBits = 8,
    uopMemDepth = 2048,
    inpMemDepth = 2048,
    wgtMemDepth = 1024,
    accMemDepth = 2048,
    outMemDepth = 2048,
    instQueueEntries = 512)
})
