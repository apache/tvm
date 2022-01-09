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
 * \file ViewController.h
 */

#import <UIKit/UIKit.h>
#import "RPCServer.h"

@interface ViewController : UIViewController <RPCServerEventListener, UITextFieldDelegate>

@property(weak, nonatomic) IBOutlet UITextField* proxyURL;
@property(weak, nonatomic) IBOutlet UITextField* proxyPort;
@property(weak, nonatomic) IBOutlet UITextField* proxyKey;
@property(weak, nonatomic) IBOutlet UILabel* statusLabel;
@property(weak, nonatomic) IBOutlet UITextView* infoText;

- (IBAction)connect:(id)sender;
@property(retain, nonatomic) IBOutlet UIButton* ConnectButton;
@property(retain, nonatomic) IBOutlet UISegmentedControl* ModeSelector;

@end
