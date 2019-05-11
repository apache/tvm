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
 * \file ViewController.h
 */

#import <UIKit/UIKit.h>
#include "TVMRuntime.h"

@interface ViewController : UIViewController<NSStreamDelegate>
{
  // input socket stream
  NSInputStream *inputStream_;
  // output socket stream
  NSOutputStream *outputStream_;
  // temporal receive buffer.
  std::string recvBuffer_;
  // Whether connection is initialized.
  bool initialized_;
  // Whether auto reconnect when a session is done.
  bool auto_reconnect_;
  // The key of the server.
  std::string key_;
  // Initial bytes to be send to remote
  std::string initBytes_;
  // Send pointer of initial bytes.
  size_t initSendPtr_;
  // Event handler.
  tvm::runtime::FEventHandler handler_;
}

@property (weak, nonatomic) IBOutlet UITextField *proxyURL;
@property (weak, nonatomic) IBOutlet UITextField *proxyPort;
@property (weak, nonatomic) IBOutlet UITextField *proxyKey;
@property (weak, nonatomic) IBOutlet UILabel *statusLabel;
@property (weak, nonatomic) IBOutlet UITextView *infoText;

- (IBAction)connect:(id)sender;
- (IBAction)disconnect:(id)sender;

@end
