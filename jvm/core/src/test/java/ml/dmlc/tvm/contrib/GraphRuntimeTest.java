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

package ml.dmlc.tvm.contrib;

import ml.dmlc.tvm.*;
import ml.dmlc.tvm.rpc.Client;
import ml.dmlc.tvm.rpc.RPCSession;
import ml.dmlc.tvm.rpc.Server;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import static org.junit.Assert.assertArrayEquals;

public class GraphRuntimeTest {
  private final Logger logger = LoggerFactory.getLogger(GraphRuntime.class);
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

    TVMContext ctx = TVMContext.cpu();
    GraphModule graph = GraphRuntime.create(graphJson, libmod, ctx);

    long[] shape = new long[]{4};
    NDArray arr = NDArray.empty(shape, ctx);
    arr.copyFrom(new float[]{1f, 2f, 3f, 4f});

    NDArray out = NDArray.empty(shape, ctx);

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
      RPCSession remote = Client.connect("localhost", port.value);
      TVMContext ctx = remote.cpu();

      remote.upload(new File(libPath));
      Module mlib = remote.loadModule("graph_addone_lib.so");

      GraphModule graph = GraphRuntime.create(graphJson, mlib, ctx);

      long[] shape = new long[]{4};
      NDArray arr = NDArray.empty(shape, ctx);
      arr.copyFrom(new float[]{1f, 2f, 3f, 4f});

      NDArray out = NDArray.empty(shape, ctx);

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
