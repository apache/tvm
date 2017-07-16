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
