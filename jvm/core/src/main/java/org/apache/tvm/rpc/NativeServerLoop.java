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

import java.io.File;
import java.io.IOException;

/**
 * Call native ServerLoop on socket file descriptor.
 */
public class NativeServerLoop implements Runnable {
  private final Function fsend;
  private final Function frecv;

  /**
   * Constructor for NativeServerLoop.
   * @param fsend socket.send function.
   * @param frecv socket.recv function.
   */
  public NativeServerLoop(final Function fsend, final Function frecv) {
    this.fsend = fsend;
    this.frecv = frecv;
  }

  @Override public void run() {
    File tempDir = null;
    try {
      tempDir = serverEnv();
      System.err.println("starting server loop...");
      RPC.getApi("ServerLoop").pushArg(fsend).pushArg(frecv).invoke();
      System.err.println("done server loop...");
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (tempDir != null) {
        String[] entries = tempDir.list();
        for (String s : entries) {
          File currentFile = new File(tempDir.getPath(), s);
          if (!currentFile.delete()) {
            System.err.println(
                "[WARN] Couldn't delete temporary file " + currentFile.getAbsolutePath());
          }
        }
        if (!tempDir.delete()) {
          System.err.println(
              "[WARN] Couldn't delete temporary directory " + tempDir.getAbsolutePath());
        }
      }
    }
  }

  private static File serverEnv() throws IOException {
    // Server environment function return temp dir.
    final File tempDir = File.createTempFile("tvm4j_rpc_", "");
    if (!tempDir.delete() || !tempDir.mkdir()) {
      throw new IOException("Couldn't create directory " + tempDir.getAbsolutePath());
    }

    Function.register("tvm.rpc.server.workpath", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        return tempDir + File.separator + args[0].asString();
      }
    }, true);

    Function.register("tvm.rpc.server.load_module", new Function.Callback() {
      @Override public Object invoke(TVMValue... args) {
        String filename = args[0].asString();
        String path = tempDir + File.separator + filename;
        System.err.println("Load module from " + path);
        return Module.load(path);
      }
    }, true);

    return tempDir;
  }
}
