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

package ml.dmlc.tvm.rpc;

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.TVMValue;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class RPCTest {
  static class RefInt {
    public int value;
  }

  private static Server startServer(RefInt portRef) {
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

  @Test
  public void test_addone() {
    if (!Module.enabled("rpc")) {
      return;
    }
    Function.register("test.rpc.addone", new Function.Callback() {
        @Override public Object invoke(TVMValue... args) {
          return args[0].asLong() + 1L;
        }
      });

    RefInt port = new RefInt();
    Server server = null;
    try {
      server = startServer(port);
      RPCSession client = Client.connect("localhost", port.value);
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
      return;
    }
    Function.register("test.rpc.strcat", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        return args[0].asString() + ":" + args[1].asLong();
      }
    });

    RefInt port = new RefInt();
    Server server = null;
    try {
      server = startServer(port);
      RPCSession client = Client.connect("localhost", port.value);
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
