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
 * \file ViewController.mm
 */

#import "ViewController.h"
#import "rpc_args.h"

@implementation ViewController

- (void)viewDidLoad {
  // To handle end editing events
  self.proxyURL.delegate = self;
  self.proxyPort.delegate = self;
  self.proxyKey.delegate = self;

  RPCArgs args = get_current_rpc_args();
  self.proxyURL.text = @(args.host_url);
  self.proxyPort.text = @(args.host_port).stringValue;
  self.proxyKey.text = @(args.key);
  
  self.ModeSelector.selectedSegmentIndex = args.server_mode;
  
  // Connect to tracker immediately
  if (args.immediate_connect) {
    [self disableUIInteraction];
    [self open];
  }
}

/*!
 * \brief Disable all UI elements
 */
- (void)disableUIInteraction {
  void (^disable)(UITextField* field) = ^(UITextField* field) {
    field.enabled = NO;
    field.backgroundColor = [UIColor lightGrayColor];
  };

  disable(self.proxyURL);
  disable(self.proxyPort);
  disable(self.proxyKey);
  self.ConnectButton.enabled = NO;
  self.DisconnectButton.enabled = NO;
  self.ModeSelector.enabled = NO;
}

/*!
 * \brief Start RPC server
 */
- (void)open {
  RPCServerMode server_mode = static_cast<RPCServerMode>(self.ModeSelector.selectedSegmentIndex);

  server_ = [RPCServer serverWithMode:server_mode];
  [server_ setDelegate:self];
  [server_ startWithHost:self.proxyURL.text
                    port:self.proxyPort.text.intValue
                     key:self.proxyKey.text];
  
  NSLog(@"Connecting to the proxy server...");
  self.infoText.text = @"";
  self.statusLabel.text = @"Connecting...";
}

/*!
 * \brief Stop RPC server
 */
- (void)close {
  [server_ stop];
  NSLog(@"Closing the streams...");
  self.statusLabel.text = @"Disconnecting...";
}

#pragma mark - Button responders

- (IBAction)connect:(id)sender {
  [[self view] endEditing:YES];
  [self open];
}

- (IBAction)disconnect:(id)sender {
  [[self view] endEditing:YES];
  [self close];
}


#pragma mark - UITextFieldDelegate

- (BOOL)textFieldShouldReturn:(UITextField *)textField {
  [[self view] endEditing:YES];  // to hide keyboard on ret key
  return FALSE;
}

- (void)textFieldDidEndEditing:(UITextField *)textField {
  // Update values in app arg cache
  RPCArgs args = get_current_rpc_args();
  args.host_url = [self.proxyURL.text UTF8String];
  args.host_port = [self.proxyPort.text intValue];
  args.key = [self.proxyKey.text UTF8String];
  set_current_rpc_args(args);
}


#pragma mark - RPCServerEvenlListener

- (void)onError:(NSString*) msg {
  dispatch_sync(dispatch_get_main_queue(), ^{
    self.infoText.text = [NSString stringWithFormat:@"Error: %@", msg];
  });
}

- (void)onStatusChanged:(RPCServerStatus) status {
  dispatch_sync(dispatch_get_main_queue(), ^{
    switch (status) {
      case RPCServerStatus_Connected:
        self.statusLabel.text = @"Connected";
        break;
      case RPCServerStatus_Disconnected:
        self.statusLabel.text = @"Disconnected";
        break;
      default:
        // Nothing
        break;
    }
  });
}

- (void)dealloc {
  [_ModeSelector release];
  [super dealloc];
}
@end
