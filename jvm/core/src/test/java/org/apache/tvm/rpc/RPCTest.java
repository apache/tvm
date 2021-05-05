/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.tvm.rpc;

import org.apache.tvm.Function;
import org.apache.tvm.Module;
import org.apache.tvm.TVMValue;
import org.apache.tvm.TestUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

public class RPCTest {
  private final Logger logger = LoggerFactory.getLogger(RPCTest.class);

  @Test
  public void test_addone() {
    if (!Module.enabled("rpc")) {
      logger.warn("RPC is not enabled. Skip.");
      return;
    }
    Function.register("test.rpc.addone", new Function.Callback() {
        @Override public Object invoke(TVMValue... args) {
          return args[0].asLong() + 1L;
        }
      });

    TestUtils.RefInt port = new TestUtils.RefInt();
    Server server = null;
    try {
      server = TestUtils.startServer(port);
      RPCSession client = Client.connect("127.0.0.1", port.value);
      Function func = client.getFunction("test.rpc.addone");
      assertEquals(11L, func.call(10).asLong());
    } finally {
      if (server != null) {
        server.terminate();
      }
    }
  }

  @Test
  public void test_strcat() {
    if (!Module.enabled("rpc")) {
      logger.warn("RPC is not enabled. Skip.");
      return;
    }
    Function.register("test.rpc.strcat", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        return args[0].asString() + ":" + args[1].asLong();
      }
    });

    TestUtils.RefInt port = new TestUtils.RefInt();
    Server server = null;
    try {
      server = TestUtils.startServer(port);
      RPCSession client = Client.connect("127.0.0.1", port.value);
      Function func = client.getFunction("test.rpc.strcat");
      assertEquals("abc:11", func.call("abc", 11L).asString());
    } finally {
      if (server != null) {
        server.terminate();
      }
    }
  }

  @Ignore("Proxy server may not have been ready when this test runs,"
        + " will add retry when callback function can deal with Java exception."
        + " After that we'll enable this test.")
  @Test
  public void test_connect_proxy_server() {
    String proxyHost = System.getProperty("test.rpc.proxy.host");
    int proxyPort = Integer.parseInt(System.getProperty("test.rpc.proxy.port"));

    Function.register("test.rpc.proxy.addone", new Function.Callback() {
      @Override public Object invoke(TVMValue... tvmValues) {
        return tvmValues[0].asLong() + 1L;
      }
    });

    Server server = null;
    try {
      server = new Server(proxyHost, proxyPort, "x1");
      server.start();

      RPCSession client = Client.connect(proxyHost, proxyPort, "x1");
      Function f1 = client.getFunction("test.rpc.proxy.addone");
      assertEquals(11L, f1.call(10L).asLong());
    } finally {
      if (server != null) {
        server.terminate();
      }
    }
  }
}
