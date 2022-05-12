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
#import "RPCArgs.h"

@implementation ViewController {
  // server implementation
  RPCServer* server_;
  // Button state. True - push will start connection, false - push will disconnect
  bool to_connect_;
}

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
  self->to_connect_ = true;

  // Add border to button
  void (^addBorder)(UIButton* btn) = ^(UIButton* btn) {
    btn.layer.borderWidth = 2.0f;
    btn.layer.borderColor = self.ConnectButton.currentTitleColor.CGColor;
    btn.layer.cornerRadius = 10;
  };
  addBorder(self.ConnectButton);

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

  void (^disableButton)(UIButton* btn) = ^(UIButton* btn) {
    btn.enabled = NO;
    btn.layer.borderColor = btn.currentTitleColor.CGColor;
  };

  disable(self.proxyURL);
  disable(self.proxyPort);
  disable(self.proxyKey);
  disableButton(self.ConnectButton);
  self.ModeSelector.enabled = NO;
}

/*!
 * \brief Start RPC server
 */
- (void)open {
  RPCArgs args = get_current_rpc_args();

  RPCServerMode server_mode = static_cast<RPCServerMode>(self.ModeSelector.selectedSegmentIndex);

  server_ = [RPCServer serverWithMode:server_mode];
  server_.host = self.proxyURL.text;
  server_.port = self.proxyPort.text.intValue;
  server_.key = self.proxyKey.text;
  server_.custom_addr = [NSString stringWithUTF8String:args.custom_addr];
  server_.verbose = args.verbose;
  server_.delegate = self;

  [server_ start];

  self.infoText.text = @"";
  self.statusLabel.text = @"Connecting...";
}

/*!
 * \brief Stop RPC server
 */
- (void)close {
  [server_ stop];
  self.statusLabel.text = @"Disconnecting...";
}

#pragma mark - Button responders
/*!
 * \brief Connect/disconnect button handler
 */
- (IBAction)connect:(id)sender {
  [[self view] endEditing:YES];  // to hide keyboard
  (to_connect_ ^= true) ? [self close] : [self open];
  [self.ConnectButton setTitle:to_connect_ ? @"Connect" : @"Disconenct"
                      forState:UIControlStateNormal];
}

#pragma mark - UITextFieldDelegate

- (BOOL)textFieldShouldReturn:(UITextField*)textField {
  [[self view] endEditing:YES];  // to hide keyboard on ret key
  return FALSE;
}

- (void)textFieldDidEndEditing:(UITextField*)textField {
  // Update values in app arg cache
  RPCArgs args = get_current_rpc_args();
  args.host_url = [self.proxyURL.text UTF8String];
  args.host_port = [self.proxyPort.text intValue];
  args.key = [self.proxyKey.text UTF8String];
  set_current_rpc_args(args);
}

#pragma mark - RPCServerEvenlListener

- (void)onError:(NSString*)msg {
  dispatch_sync(dispatch_get_main_queue(), ^{
    self.infoText.text = [NSString stringWithFormat:@"Error: %@", msg];
  });
}

- (void)onStatusChanged:(RPCServerStatus)status {
  dispatch_sync(dispatch_get_main_queue(), ^{
    switch (status) {
      case RPCServerStatus_Connected:
        if (self.ModeSelector.selectedSegmentIndex == RPCServerMode_Standalone) {
          self.infoText.text = [NSString
              stringWithFormat:@"IP: %@\nPort: %d", server_.device_addr, server_.actual_port];
        }
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

@end
