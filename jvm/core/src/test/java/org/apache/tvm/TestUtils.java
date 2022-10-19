package org.apache.tvm;

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


import org.apache.tvm.rpc.Server;

import java.io.IOException;

public class TestUtils {
  public static class RefInt {
    public int value;
  }

  public static Server startServer(RefInt portRef) {
    Server server = null;
    int port = 9981;
    for (int i = 0; i < 10; ++i) {
      try {
        server = new Server(port + i);
        server.start();
        portRef.value = port + i;
        return server;
      } catch (IOException e) {
      }
    }
    throw new RuntimeException("Cannot find an available port.");
  }
}
