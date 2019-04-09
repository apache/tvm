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
 *  Copyright (c) 2017 by Contributors
 * \file TVMRuntime.h
 */
#import <Foundation/Foundation.h>
// Customize logging mechanism, redirect to NSLOG
#define DMLC_LOG_CUSTOMIZE 1
#define TVM_METAL_RUNTIME 1

#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <functional>

namespace tvm {
namespace runtime {

/*!
 * \brief Message handling function for event driven server.
 *
 * \param in_bytes The incoming bytes.
 * \param event_flag  1: read_available, 2: write_avaiable.
 * \return State flag.
 *     1: continue running, no need to write,
 *     2: need to write
 *     0: shutdown
 */
using FEventHandler = std::function<int(const std::string& in_bytes, int event_flag)>;

/*!
 * \brief Create a server event handler.
 *
 * \param outputStream The output stream used to send outputs.
 * \param name The name of the server.
 * \param remote_key The remote key
 * \return The event handler.
 */
FEventHandler CreateServerEventHandler(NSOutputStream *outputStream,
                                       std::string name,
                                       std::string remote_key);

}  // namespace runtime
}  // namespace tvm

@interface TVMRuntime : NSObject

+ (void)launchSyncServer;

@end
