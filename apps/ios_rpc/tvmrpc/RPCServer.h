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
 * \file Provide interfaces to launch and control RPC Service routine
 */

#import <Foundation/Foundation.h>

/*!
 * \brief Enum with possible status of RPC server
 * Used to report state to listener
 */
typedef enum {
  RPCServerStatus_Launched,           // Worker thread is launched
  RPCServerStatus_Stopped,            // Worker thread stopped
  RPCServerStatus_Connected,          // Connected to Proxy/Tracker
  RPCServerStatus_Disconnected,       // Disconnected from Proxy/Tracker
  RPCServerStatus_RPCSessionStarted,  // RPC session is started
  RPCServerStatus_RPCSessionFinished  // RPC session is finished
} RPCServerStatus;

/*!
 * \brief Enum with modes of servicing supported by RPCServer
 */
typedef enum {
  /// Tracker mode. Same as Standalone Server plus register it into Tracker.
  RPCServerMode_Tracker,
  /// Proxy mode. Connect to proxy server and wait response.
  RPCServerMode_Proxy,
  /// Standalone RPC server mode. Open port with RPC server and wait incoming connection.
  RPCServerMode_Standalone
} RPCServerMode;

/*!
 * \brief Listener for events happened with RPCServer
 */
@protocol RPCServerEventListener <NSObject>
/// Callback to notifying about new status
- (void)onError:(NSString*)msg;
/// Callback to notifying about error
- (void)onStatusChanged:(RPCServerStatus)status;
@end

/*!
 * \brief RPC Server instance
 * Contains internal worker thread plus
 */
@interface RPCServer : NSObject <NSStreamDelegate>

/// Event listener delegate to set
@property(retain) id<RPCServerEventListener> delegate;
/// Device key to report during RPC session
@property(retain) NSString* key;
/// Host address of Proxy/Tracker server (generally IPv4). Ignored for Standalone mode.
@property(retain) NSString* host;
/// Port of Proxy/Tracker server. Ignored for Standalone mode.
@property int port;
/// Custom address to report into tracker server (optional). Ignored for Standalone/Proxy modes
@property(retain) NSString* custom_addr;
/// Triger to enable printing of server state info
@property BOOL verbose;
/// RPC port opened on the device. Ignored for Proxy/Tracker modes
@property int actual_port;
/// IP address of the device. Ignored for Proxy/Tracker modes
@property(retain) NSString* device_addr;

/*!
 * \brief Create server with specified sevicing mode
 * \param mode Mode of server
 */
+ (instancetype)serverWithMode:(RPCServerMode)mode;

/*!
 * \brief Start RPC server with options. Non blocking method
 */
- (void)start;

/*!
 * \brief Stop RPC server. Non blocking method
 */
- (void)stop;

@end
