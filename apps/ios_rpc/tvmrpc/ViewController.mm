/*!
 *  Copyright (c) 2017 by Contributors
 * \file ViewController.mm
 */

#include <string>
#import "ViewController.h"

@implementation ViewController

- (void)stream:(NSStream *)strm handleEvent:(NSStreamEvent)event {
  std::string buffer;
  switch (event) {
    case NSStreamEventOpenCompleted: {
      self.statusLabel.text = @"Connected";
      break;
    }
    case NSStreamEventHasBytesAvailable:
      if (strm == inputStream_) {
        [self onReadAvailable];
      }
      break;
    case NSStreamEventHasSpaceAvailable: {
      if (strm == outputStream_) {
        [self onWriteAvailable];
      }
      break;
    }
    case NSStreamEventErrorOccurred: {
      NSLog(@"%@",[strm streamError].localizedDescription);
      break;
    }
    case NSStreamEventEndEncountered: {
      [self close];
      // auto reconnect when normal end.
      [self open];
      break;
    }
    default: {
      NSLog(@"Unknown event");
    }
  }
}

- (void)onReadAvailable {
  constexpr int kRPCMagic = 0xff271;
  if (!initialized_) {
    int code;
    size_t nbytes = [inputStream_ read:reinterpret_cast<uint8_t*>(&code)
                                  maxLength:sizeof(code)];
    if (nbytes != sizeof(code)) {
      self.infoText.text = @"Fail to receive remote confirmation code.";
      [self close];
    } else if (code == kRPCMagic + 2) {
      self.infoText.text = @"Proxy server cannot find client that matches the key";
      [self close];
    } else if (code == kRPCMagic + 1) {
      self.infoText.text = @"Proxy server already have another server with same key";
      [self close];
    } else if (code != kRPCMagic) {
      self.infoText.text = @"Given address is not a TVM RPC Proxy";
      [self close];
    } else {
      initialized_ = true;
      self.statusLabel.text = @"Proxy connected.";
      CHECK(handler_ != nullptr);
    }
  }
  const int kBufferSize = 4 << 10;
  if (initialized_) {
    while ([inputStream_ hasBytesAvailable]) {
      recvBuffer_.resize(kBufferSize);
      uint8_t* bptr = reinterpret_cast<uint8_t*>(&recvBuffer_[0]);
      size_t nbytes = [inputStream_ read:bptr maxLength:kBufferSize];
      recvBuffer_.resize(nbytes);
      int flag = 1;
      if ([outputStream_ hasSpaceAvailable]) {
        flag |= 2;
      }
      // always try to write
      try {
        flag = handler_(recvBuffer_, flag);
        if (flag == 2) {
          [self onShutdownReceived];
        }
      } catch (const dmlc::Error& e) {
        [self close];
      }
    }
  }
}

- (void)onShutdownReceived {
  [self close];
}

- (void)onWriteAvailable {
  if (initSendPtr_ < initBytes_.length()) {
    initSendPtr_ += [outputStream_ write:reinterpret_cast<uint8_t*>(&initBytes_[initSendPtr_])
                                   maxLength:(initBytes_.length() - initSendPtr_)];
  }
  if (initialized_) {
    try {
      std::string dummy;
      int flag = handler_(dummy, 2);
      if (flag == 2) {
        [self onShutdownReceived];
      }
    } catch (const dmlc::Error& e) {
      [self close];
    }
  }
}

- (void)open {
  constexpr int kRPCMagic = 0xff271;
  NSLog(@"Connecting to the proxy server..");
  // Initialize the data states.
  key_ = [self.proxyKey.text UTF8String];
  key_ = "server:" + key_;
  std::ostringstream os;
  int rpc_magic = kRPCMagic;
  os.write(reinterpret_cast<char*>(&rpc_magic), sizeof(rpc_magic));
  int keylen = static_cast<int>(key_.length());
  os.write(reinterpret_cast<char*>(&keylen), sizeof(keylen));
  os.write(key_.c_str(), key_.length());
  initialized_ = false;
  initBytes_ = os.str();
  initSendPtr_ = 0;
  // Initialize the network.
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;
  CFStreamCreatePairWithSocketToHost(
      NULL,
      (__bridge CFStringRef) self.proxyURL.text,
      [self.proxyPort.text intValue],
      &readStream, &writeStream);
  inputStream_ = (__bridge_transfer NSInputStream *)readStream;
  outputStream_ = (__bridge_transfer NSOutputStream *)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];
  handler_ = tvm::runtime::CreateServerEventHandler(outputStream_, key_, "%toinit");
  CHECK(handler_ != nullptr);
  self.infoText.text = @"";
  self.statusLabel.text = @"Connecting...";
}

- (void)close {
  NSLog(@"Closing the streams.");
  [inputStream_ close];
  [outputStream_ close];
  [inputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [inputStream_ setDelegate:nil];
  [outputStream_ setDelegate:nil];
  inputStream_ = nil;
  outputStream_ = nil;
  handler_ = nullptr;
  self.statusLabel.text = @"Disconnected";
}

- (IBAction)connect:(id)sender {
  [self open];
  [[self view] endEditing:YES];
}

- (IBAction)disconnect:(id)sender {
  [self close];
}

@end
