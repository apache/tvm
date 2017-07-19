/*!
 *  Copyright (c) 2017 by Contributors
 * \brief A hook to launch RPC server via xcodebuild test
 * \file tvmrpcLauncher.mm
 */

#import <XCTest/XCTest.h>
#import "TVMRuntime.h"

@interface tvmrpcLauncher : XCTestCase

@end

@implementation tvmrpcLauncher

- (void)setUp {
    [super setUp];
}

- (void)tearDown {
    [super tearDown];
}

- (void)testRPC {
  [TVMRuntime launchSyncServer];
}


@end
