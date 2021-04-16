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

#import "RPCServer.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <random>

// To get device WiFi IP
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// TVM internal header to access Magic keys like kRPCMagic and others
#include "../../../src/runtime/rpc/rpc_endpoint.h"

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
using FEventHandler = PackedFunc;

/*!
 * \brief Create a server event handler.
 *
 * \param outputStream The output stream used to send outputs.
 * \param name The name of the server.
 * \param remote_key The remote key
 * \return The event handler.
 */
FEventHandler CreateServerEventHandler(NSOutputStream* outputStream, std::string name,
                                       std::string remote_key) {
  const PackedFunc* event_handler_factor = Registry::Get("rpc.CreateEventDrivenServer");
  ICHECK(event_handler_factor != nullptr)
    << "You are using tvm_runtime module built without RPC support. "
    << "Please rebuild it with USE_RPC flag.";

  PackedFunc writer_func([outputStream](TVMArgs args, TVMRetValue* rv) {
    TVMByteArray *data = args[0].ptr<TVMByteArray>();
    int64_t nbytes = [outputStream write:reinterpret_cast<const uint8_t*>(data->data) maxLength:data->size];
    if (nbytes < 0) {
      NSLog(@"%@", [outputStream streamError].localizedDescription);
      throw tvm::Error("Stream error");
    }
    *rv = nbytes;
  });

  return (*event_handler_factor)(writer_func, name, remote_key);
}

/*!
 * \brief Helper function to query real IP of device in WiFi network
 * \return string with IPv4 in format "192.168.0.1" or "unknown" if cannot detect
 */
static std::string getWiFiAddress() {
  std::string address = "unknown";
  ifaddrs *interfaces = nullptr;

  int success = getifaddrs(&interfaces);
  if (success == 0) {
      // Loop through linked list of interfaces
      ifaddrs *temp_addr = interfaces;
      while(temp_addr != NULL) {
          if(temp_addr->ifa_addr->sa_family == AF_INET) {
              // Check if interface is en0 which is the wifi connection on the iPhone
              if(std::string(temp_addr->ifa_name) == "en0") {
                  address = std::string(inet_ntoa(((sockaddr_in *)temp_addr->ifa_addr)->sin_addr));
              }
          }
          temp_addr = temp_addr->ifa_next;
      }
  }

  freeifaddrs(interfaces);
  return address;
}

}  // namespace runtime
}  // namespace tvm


// Forward declaration
@interface RPCTrackerClien : RPCServer<RPCServerEventListener>
@end

// Forward declaration
@interface RPCServerPure : RPCServer
@end

/*!
 * \brief Base implementation of server to work with RPCProxy
 *        Will automatically connect/reconnect to RPCProxy server
 */
@implementation RPCServer {
@protected
  // The key of the server.
  NSString* key_;
  // The url of host.
  NSString* url_;
  // The port of host.
  NSInteger port_;
  // Event listener
  id <RPCServerEventListener> delegate_;
  // Worker thread
  NSThread* worker_thread_;
  // Triger to continue processing
  BOOL shouldKeepRunning;
  // Ip of rpc server (actually ip of ios device)
  std::string server_ip_;
  // Port of rpc server
  int server_port_;
  // Input socket stream
  NSInputStream* inputStream_;
  // Output socket stream
  NSOutputStream* outputStream_;
  // Temporal receive buffer.
  std::string recvBuffer_;
  // Whether connection is initialized.
  bool initialized_;
  // Initial bytes to be send to remote
  std::string initBytes_;
  // Send pointer of initial bytes.
  size_t initSendPtr_;
  // Event handler.
  tvm::runtime::FEventHandler handler_;
}

+ (instancetype)serverWithMode:(RPCServerMode) mode {
  if (mode == RPCServerMode_PureServer)
    return [[RPCServerPure alloc] init];
  if (mode == RPCServerMode_Proxy)
    return [[RPCServer alloc] init];
  if (mode == RPCServerMode_Tracker)
    return [[RPCTrackerClien alloc] init];
  return nil;
}

- (instancetype)init {
  [super init];
  server_ip_ = tvm::runtime::getWiFiAddress();
  return self;
}

/*!
 * Internal setters methods
 */
- (void)setDelegate:(id<RPCServerEventListener>) delegate { delegate_ = delegate; }
- (void)setKey:(NSString*) key { key_ = key; }
- (void)setUrl:(NSString*) url { url_ = url; }
- (void)setPort:(NSInteger) port { port_ = port; }

/*!
 * \brief Main event listener method. All stream event handling starts here
 */
- (void)stream:(NSStream*)strm handleEvent:(NSStreamEvent)event {
  std::string buffer;
  switch (event) {
    case NSStreamEventOpenCompleted: {
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
      NSLog(@"%@", [strm streamError].localizedDescription);
      break;
    }
    case NSStreamEventEndEncountered: {
      [self onEndEvent];
      break;
    }
    default: {
      NSLog(@"Unknown event");
    }
  }
}

-(void)notifyError:(NSString*) msg {
  NSLog(@"[IOS-RPC] ERROR: %@", msg);
  if (delegate_ != nil)
    [delegate_ onError:msg];
}

-(void)notifyState:(RPCServerStatus) state {
  if (state == RPCServerStatus_Launched) {
    // Notify host runner script with actual address
    NSLog(@"[IOS-RPC] IP: %s", server_ip_.c_str());
    NSLog(@"[IOS-RPC] PORT: %d", server_port_);
  }
  // Notify host runner script with current status
  NSLog(@"[IOS-RPC] STATE: %d", state);
  // Notify listener
  if (delegate_ != nil)
    [delegate_ onStatusChanged:state];
}

- (void)onReadAvailable {
  if (!initialized_) {
    using tvm::runtime::kRPCMagic;
    int code;
    size_t nbytes = [inputStream_ read:reinterpret_cast<uint8_t*>(&code) maxLength:sizeof(code)];
    if (nbytes != sizeof(code)) {
      [self notifyError:@"Fail to receive remote confirmation code."];
      [self close];
    } else if (code == kRPCMagic + 2) {
      [self notifyError:@"Proxy server cannot find client that matches the key"];
      [self close];
    } else if (code == kRPCMagic + 1) {
      [self notifyError:@"Proxy server already have another server with same key"];
      [self close];
    } else if (code != kRPCMagic) {
      [self notifyError:@"Given address is not a TVM RPC Proxy"];
      [self close];
    } else {
      initialized_ = true;
      ICHECK(handler_ != nullptr);
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
        TVMByteArray arr {recvBuffer_.data(), recvBuffer_.size()};
        flag = handler_(arr, flag);
        if (flag == 2) {
          [self onShutdownReceived];
        }
      } catch (const tvm::Error& e) {
        [self close];
      }
    }
  }
}

- (void)onWriteAvailable {
  if (initSendPtr_ < initBytes_.length()) {
    initSendPtr_ += [outputStream_ write:reinterpret_cast<uint8_t*>(&initBytes_[initSendPtr_])
                               maxLength:(initBytes_.length() - initSendPtr_)];
  }
  [self notifyState:RPCServerStatus_Connected];
  if (initialized_) {
    try {
      TVMByteArray dummy {nullptr, 0};
      int flag = handler_(dummy, 2);
      if (flag == 2) {
        [self onShutdownReceived];
      }
    } catch (const tvm::Error& e) {
      [self close];
    }
  }
}

- (void)onShutdownReceived {
  [self close];
}

- (void)onEndEvent {
  [self close];
  [self notifyState:RPCServerStatus_RPCSessionFinished];
  // Basic behaviour is to reconnect
  [self open];
}

- (void)open {
  // Initialize the data states.
  std::string full_key = std::string("server:") + [key_ UTF8String];
  std::ostringstream os;
  int rpc_magic = tvm::runtime::kRPCMagic;
  os.write(reinterpret_cast<char*>(&rpc_magic), sizeof(rpc_magic));
  int keylen = static_cast<int>(full_key.length());
  os.write(reinterpret_cast<char*>(&keylen), sizeof(keylen));
  os.write(full_key.c_str(), full_key.length());
  initialized_ = false;
  initBytes_ = os.str();
  initSendPtr_ = 0;
  
  // Initialize the network.
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;
  CFStreamCreatePairWithSocketToHost(NULL, (CFStringRef)url_, port_,
                                     &readStream, &writeStream);
  inputStream_ = (NSInputStream*)readStream;
  outputStream_ = (NSOutputStream*)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];

  handler_ = tvm::runtime::CreateServerEventHandler(outputStream_, full_key, "%toinit");
  ICHECK(handler_ != nullptr);
}

- (void)close {
  [inputStream_ close];
  [outputStream_ close];
  [inputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [inputStream_ setDelegate:nil];
  [outputStream_ setDelegate:nil];
  inputStream_ = nil;
  outputStream_ = nil;
  handler_ = nullptr;
  [self notifyState:RPCServerStatus_Disconnected];
}

- (void)startWithHost:(NSString*) host port: (int) port key:(NSString*) key {
  key_ = [key copy];
  port_ = port;
  url_ = [host copy];
  
  // process in separate thead with runloop
  worker_thread_ = [[NSThread alloc] initWithBlock:^{
    @autoreleasepool {
      [self open];
      [self notifyState:RPCServerStatus_Launched];
      shouldKeepRunning = YES;
      while (shouldKeepRunning && [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate distantFuture]]);
      [self notifyState:RPCServerStatus_Stopped];
    }
  }];
  [worker_thread_ start];
}

- (void)stop {
  if (worker_thread_ == nil)
    return;

  [self performSelector:@selector(stop_) onThread:worker_thread_ withObject:nil waitUntilDone:NO];
  worker_thread_ = nil;
}

- (void)stop_ {
  [self close];
  shouldKeepRunning = NO;
}

@end

@implementation RPCServerPure {
  // Socket with incoming conenction (only for Pure RPC server)
  CFSocketNativeHandle socket_;
}

- (void)handleConnect:(const CFSocketNativeHandle) hdl {
  socket_ = hdl;
  
  // Initialize the data states.
  std::string full_key = std::string("server:") + [key_ UTF8String];
  std::ostringstream os;
  int rpc_magic = tvm::runtime::kRPCMagic;
  os.write(reinterpret_cast<char*>(&rpc_magic), sizeof(rpc_magic));
  int keylen = static_cast<int>(full_key.length());
  os.write(reinterpret_cast<char*>(&keylen), sizeof(keylen));
  os.write(full_key.c_str(), full_key.length());
  initialized_ = false;
  initBytes_ = os.str();
  initSendPtr_ = 0;
  
  // Initialize the network.
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;
  CFStreamCreatePairWithSocket(NULL, socket_, &readStream, &writeStream);

  inputStream_ = (NSInputStream*)readStream;
  outputStream_ = (NSOutputStream*)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];

  handler_ = tvm::runtime::CreateServerEventHandler(outputStream_, full_key, "%toinit");
  ICHECK(handler_ != nullptr);
}

static void handleConnect(CFSocketRef socket, CFSocketCallBackType type, CFDataRef address, const void *data, void *info) {
  RPCServerPure* it = static_cast<RPCServerPure*>(info);
  [it handleConnect:*static_cast<const CFSocketNativeHandle*>(data)];
}

- (void)open {
  CFSocketContext ctx {};
  ctx.info = self;

  CFSocketRef myipv4cfsock = CFSocketCreate(
      kCFAllocatorDefault, PF_INET, SOCK_STREAM, IPPROTO_TCP,
      kCFSocketAcceptCallBack, handleConnect, &ctx);

  struct sockaddr_in sin;
  int rpc_port = 9090;  // TODO: hardcoded. Should try bind in range of ports
  memset(&sin, 0, sizeof(sin));
  sin.sin_len = sizeof(sin);
  sin.sin_family = AF_INET;
  sin.sin_port = htons(rpc_port);
  sin.sin_addr.s_addr= INADDR_ANY;

  CFDataRef sincfd = CFDataCreate(
      kCFAllocatorDefault,
      (UInt8 *)&sin,
      sizeof(sin));

  if (CFSocketSetAddress(myipv4cfsock, sincfd) != 0)
    @throw [NSException exceptionWithName:@"SocketError"
                                   reason:[NSString stringWithFormat:@"Can not bind to port %d", rpc_port]
                                 userInfo:nil];
  CFRelease(sincfd);
  server_port_ = rpc_port;

  CFRunLoopSourceRef socketsource = CFSocketCreateRunLoopSource(kCFAllocatorDefault, myipv4cfsock, 0);
  CFRunLoopAddSource(CFRunLoopGetCurrent(), socketsource, kCFRunLoopDefaultMode);
}

- (void)onEndEvent {
  [self close];
  [self notifyState:RPCServerStatus_RPCSessionFinished];
}

@end

typedef enum {
  HandshakeToSend,
  HandshakeToRecv,
  ServerInfoToSend,
  ServerInfoToRecv,
  ReportResToSend,
  ReportResToRecv,
  UpdateKeyToSend,
  UpdateKeyToRecv,
  WaitConnection,
} TrackerClientState;

@implementation RPCTrackerClien {
  // RPC Server to register in tracker
  RPCServerPure* rpc_server_;
  // Size of data required accumulate in readBuffer before processing
  int required_data_size;
  // State of tracker client
  TrackerClientState state_;
}

- (void)toSend:(NSData*) data {
  // try to send
  NSInteger sent_size = [outputStream_ write:(uint8_t*)data.bytes  maxLength:data.length];
  // assume that all data is sent
  if (sent_size != data.length)
    @throw [NSException exceptionWithName:@"SocketError"
                                   reason:[NSString stringWithFormat:@"Unable to send data"]
                                 userInfo:nil];
}

- (void)toSendPacked:(NSData*) data {
  // try to send
  int packet_size = data.length;
  NSInteger sent_size = [outputStream_ write:(uint8_t*)&packet_size  maxLength:sizeof(packet_size)];
  if (sent_size != sizeof(packet_size))
    @throw [NSException exceptionWithName:@"SocketError"
                                   reason:[NSString stringWithFormat:@"Unable to send data"]
                                 userInfo:nil];
  
  NSInteger sent_data = [outputStream_ write:(uint8_t*)data.bytes  maxLength:data.length];
  // assume that all data is sent
  if (sent_data != data.length)
    @throw [NSException exceptionWithName:@"SocketError"
                                   reason:[NSString stringWithFormat:@"Unable to send data"]
                                 userInfo:nil];
}

- (void)onReadAvailable {
  const int kBufferSize = 4 << 10;
  const int prev_size = recvBuffer_.size();
  recvBuffer_.resize(kBufferSize);
  size_t nbytes = [inputStream_ read:(uint8_t*)recvBuffer_.data() + prev_size
                           maxLength:recvBuffer_.size() - prev_size];
  recvBuffer_.resize(nbytes + prev_size);
  
  if (recvBuffer_.size() < required_data_size)
    return;
  
  switch (state_) {
    case HandshakeToRecv: {
      int code = tvm::runtime::kRPCTrackerMagic;
      if (recvBuffer_.size() < sizeof(code)) {
        required_data_size = sizeof(code);
        break;
      }
      
      if (recvBuffer_.size() != sizeof(code) || *(int*)recvBuffer_.data() != code) {
        [self notifyError:@"Wrong responce, server is not tracker."];
        [self close];
        break;
      }
      
      recvBuffer_.erase(recvBuffer_.begin(), recvBuffer_.begin() + sizeof(code));
      required_data_size = 0;
      [self notifyState:RPCServerStatus_Connected];
      state_ = ServerInfoToSend;
      break;
    }
    case ServerInfoToRecv: {
      std::string expected = std::to_string(static_cast<int>(tvm::runtime::TrackerCode::kSuccess));
      int packet_size;
      if (recvBuffer_.size() < sizeof(packet_size)) {
        required_data_size = sizeof(packet_size);
        break;
      }
      
      packet_size = *(int*)recvBuffer_.data();
      if (recvBuffer_.size() < sizeof(packet_size) + packet_size) {
        required_data_size = sizeof(packet_size) + packet_size;
        break;
      }

      if (std::string((char*)recvBuffer_.data() + sizeof(packet_size), packet_size) != expected) {
        [self notifyError:@"Was not able to update info in tracker. Responce is not OK."];
        [self close];
        break;
      }
      
      recvBuffer_.erase(recvBuffer_.begin(), recvBuffer_.begin() + sizeof(packet_size) + packet_size);
      required_data_size = 0;
      state_ = ReportResToSend;
      break;
    }
    case ReportResToRecv: {
      std::string expected = std::to_string(static_cast<int>(tvm::runtime::TrackerCode::kSuccess));
      int packet_size;
      if (recvBuffer_.size() < sizeof(packet_size)) {
        required_data_size = sizeof(packet_size);
        break;
      }
      
      packet_size = *(int*)recvBuffer_.data();
      if (recvBuffer_.size() < sizeof(packet_size) + packet_size) {
        required_data_size = sizeof(packet_size) + packet_size;
        break;
      }

      if (std::string((char*)recvBuffer_.data() + sizeof(packet_size), packet_size) != expected) {
        [self notifyError:@"Was not able to put resource into tracker. Responce is not OK."];
        [self close];
        break;
      }
      
      recvBuffer_.erase(recvBuffer_.begin(), recvBuffer_.begin() + sizeof(packet_size) + packet_size);
      required_data_size = 0;
      state_ = WaitConnection;
      break;
    }
    default:
      // Nothing
      break;
  }

  if (outputStream_.hasSpaceAvailable)
    [self onWriteAvailable];
}

- (void)onWriteAvailable {
  switch (state_) {
    case HandshakeToSend: {
      int code = tvm::runtime::kRPCTrackerMagic;
      NSData* data = [NSData dataWithBytes:&code length:sizeof(code)];
      [self toSend:data];
      state_ = HandshakeToRecv;
      break;
    }
    case ServerInfoToSend: {
      std::ostringstream ss;
      ss << "[" << static_cast<int>(tvm::runtime::TrackerCode::kUpdateInfo) << ", {\"key\": \"server:" << key_.UTF8String
         << "\"}]";
      std::string data_s = ss.str();
      NSData* data = [NSData dataWithBytes:data_s.data() length:data_s.length()];
      [self toSendPacked:data];
      state_ = ServerInfoToRecv;
      break;
    }
    case ReportResToSend: {
      std::mt19937 gen(std::random_device{}());
      std::uniform_real_distribution<float> dis(0.0, 1.0);
      
      // TODO: All values are hardcoded
      int port = 9090;
      std::string custom_addr = "null";
      std::string matchkey = std::string(key_.UTF8String) + ":" + std::to_string(dis(gen));
      
      std::ostringstream ss;
      ss << "[" << static_cast<int>(tvm::runtime::TrackerCode::kPut) << ", \"" << key_.UTF8String << "\", [" << port
         << ", \"" << matchkey << "\"], " << custom_addr << "]";

      std::string data_s = ss.str();
      NSData* data = [NSData dataWithBytes:data_s.data() length:data_s.length()];
      [self toSendPacked:data];
      state_ = ReportResToRecv;
      break;
    }
    default:
      // Nothing
      break;
  }
  
  if (inputStream_.hasBytesAvailable)
    [self onReadAvailable];
}

-(void)open {
  // create RPC pure server
  //   * set self as delegate (to register back when servicing is finished)
  //   * mute printing status
  //   * stop processing by sending stop
  rpc_server_ = [[RPCServerPure alloc] init];
  rpc_server_.key = key_;
  rpc_server_.delegate = self;
  [rpc_server_ open];
  
  // Initialize the network.
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;

  CFStreamCreatePairWithSocketToHost(NULL, (CFStringRef)url_, port_,
                                     &readStream, &writeStream);
  
  inputStream_ = (NSInputStream*)readStream;
  outputStream_ = (NSOutputStream*)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];
}

- (void)close {
  [rpc_server_ close];
  [super close];
  [self notifyState:RPCServerStatus_Disconnected];
}

- (void)onEndEvent {
  [self close];
}

- (void)onError:(NSString*) msg {
  // transfer error form rpc_server_ to real delegate
  [self notifyError:msg];
}

- (void)onStatusChanged:(RPCServerStatus) status {
  if (status == RPCServerStatus_RPCSessionFinished) {
    [self notifyState:status];
    state_ = ReportResToSend;
    [self onWriteAvailable];
  }
}

@end
