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
 * \brief A hook to launch RPC server via xcodebuild test
 * \file tvmrpcLauncher.mm
 */

#import <XCTest/XCTest.h>
#import "RPCArgs.h"
#import "RPCServer.h"

@interface EventListenerWrapper : NSObject <RPCServerEventListener>
@end

@implementation EventListenerWrapper {
  void (^blk_)(RPCServerStatus);
}

- (instancetype)initWithBlock:(void (^)(RPCServerStatus))blk {
  blk_ = blk;
  return self;
}

- (void)onError:(NSString*)msg {
}
- (void)onStatusChanged:(RPCServerStatus)status {
  blk_(status);
}
// void (^disable)(UITextField* field) = ^(UITextField* field) {
@end

@interface tvmrpcLauncher : XCTestCase
@end

@implementation tvmrpcLauncher

- (void)testRPC {
  RPCArgs args = get_current_rpc_args();

  RPCServer* server = [RPCServer serverWithMode:static_cast<RPCServerMode>(args.server_mode)];
  server.host = @(args.host_url);
  server.port = args.host_port;
  server.key = @(args.key);

  server.delegate = [[EventListenerWrapper alloc] initWithBlock:^(RPCServerStatus sts) {
    NSLog(@"Sts : %d", sts);
  }];

  [server start];
}

@end
