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

package org.apache.tvm.contrib;

import org.apache.tvm.Module;
import org.apache.tvm.NDArray;
import org.apache.tvm.Device;
import org.apache.tvm.TestUtils;
import org.apache.tvm.rpc.Client;
import org.apache.tvm.rpc.RPCSession;
import org.apache.tvm.rpc.Server;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import static org.junit.Assert.assertArrayEquals;

public class GraphExecutorTest {
  private final Logger logger = LoggerFactory.getLogger(GraphExecutor.class);
  private static String loadingDir;

  @BeforeClass
  public static void beforeClass() {
    loadingDir = System.getProperty("test.tempdir");
  }

  @Test
  public void test_add_one_local() throws IOException {
    Module libmod = Module.load(loadingDir + File.separator + "graph_addone_lib.so");
    String graphJson = new Scanner(new File(
        loadingDir + File.separator + "graph_addone.json"))
        .useDelimiter("\\Z").next();

    Device dev = Device.cpu();
    GraphModule graph = GraphExecutor.create(graphJson, libmod, dev);

    long[] shape = new long[]{4};
    NDArray arr = NDArray.empty(shape, dev);
    arr.copyFrom(new float[]{1f, 2f, 3f, 4f});

    NDArray out = NDArray.empty(shape, dev);

    graph.setInput("x", arr).run();
    graph.getOutput(0, out);

    assertArrayEquals(new float[]{2f, 3f, 4f, 5f}, out.asFloatArray(), 1e-3f);

    arr.release();
    out.release();
    graph.release();
  }

  @Test
  public void test_add_one_remote() throws IOException {
    if (!Module.enabled("rpc")) {
      logger.warn("RPC is not enabled. Skip.");
      return;
    }

    String libPath = loadingDir + File.separator + "graph_addone_lib.so";
    String graphJson = new Scanner(new File(
        loadingDir + File.separator + "graph_addone.json"))
        .useDelimiter("\\Z").next();

    TestUtils.RefInt port = new TestUtils.RefInt();
    Server server = null;
    try {
      server = TestUtils.startServer(port);
      RPCSession remote = Client.connect("127.0.0.1", port.value);
      Device dev = remote.cpu();

      remote.upload(new File(libPath));
      Module mlib = remote.loadModule("graph_addone_lib.so");

      GraphModule graph = GraphExecutor.create(graphJson, mlib, dev);

      long[] shape = new long[]{4};
      NDArray arr = NDArray.empty(shape, dev);
      arr.copyFrom(new float[]{1f, 2f, 3f, 4f});

      NDArray out = NDArray.empty(shape, dev);

      graph.setInput("x", arr).run();
      graph.getOutput(0, out);

      assertArrayEquals(new float[]{2f, 3f, 4f, 5f}, out.asFloatArray(), 1e-3f);

      arr.release();
      out.release();
      graph.release();
    } finally {
      if (server != null) {
        server.terminate();
      }
    }
  }
}
