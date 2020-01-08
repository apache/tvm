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

import java.io.IOException;

/**
 * RPC Server.
 */
public class Server {
  private final WorkerThread worker;

  private static class WorkerThread extends Thread {
    private volatile boolean running = true;
    private final ServerProcessor processor;

    public WorkerThread(ServerProcessor processor) {
      this.processor = processor;
    }

    @Override public void run() {
      while (running) {
        processor.run();
        try {
          Thread.sleep(1000);
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
      }
    }

    public void terminate() {
      running = false;
      processor.terminate();
    }
  }

  /**
   * Start a standalone server.
   * @param serverPort Port.
   * @throws IOException if failed to bind localhost:port.
   */
  public Server(int serverPort) throws IOException {
    worker = new WorkerThread(new StandaloneServerProcessor(serverPort));
  }

  /**
   * Start a server connected to proxy.
   * Use sun.misc.SharedSecrets.getJavaIOFileDescriptorAccess
   * to get file descriptor for the socket.
   * @param proxyHost The proxy server host.
   * @param proxyPort The proxy server port.
   * @param key The key to identify the server.
   */
  public Server(String proxyHost, int proxyPort, String key) {
    worker = new WorkerThread(
        new ConnectProxyServerProcessor(proxyHost, proxyPort, key));
  }

  /**
   * Start the server.
   */
  public void start() {
    worker.start();
  }

  /**
   * Stop the server.
   */
  public void terminate() {
    worker.terminate();
  }
}
