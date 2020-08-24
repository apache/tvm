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

package org.apache.tvm.tvmrpc;

import android.app.Activity;
import android.os.ParcelFileDescriptor;
import java.net.Socket;
import org.apache.tvm.rpc.ConnectTrackerServerProcessor;

/**
 * Connect to RPC proxy and deal with requests.
 */
class RPCProcessor extends Thread {
  private String host;
  private int port;
  private String key;
  private boolean running = false;
  private long startTime;
  private ConnectTrackerServerProcessor currProcessor;
  private boolean first = true;
  private Activity rpc_activity = null;

  public RPCProcessor(Activity activity) {
    super();
    rpc_activity = activity;
  }

  @Override public void run() {
    RPCAndroidWatchdog watchdog = new RPCAndroidWatchdog(rpc_activity);
    watchdog.start();
    while (true) {
      synchronized (this) {
        currProcessor = null;
        while (!running) {
          try {
            this.wait();
          } catch (InterruptedException e) {
          }
        }
        try {
          currProcessor = new ConnectTrackerServerProcessor(host, port, key, watchdog);
        } catch (Throwable e) {
          e.printStackTrace();
          // kill if creating a new processor failed
          System.exit(0);
        }
      }
      if (currProcessor != null)
        currProcessor.run();
      watchdog.finishTimeout();
    }
  }

  /**
   * Disconnect from the proxy server.
   */
  synchronized void disconnect() {
    if (running) {
      running = false;
      if (currProcessor != null) {
        currProcessor.terminate();
      }
    }
  }

  /**
   * Start rpc processor and connect to the proxy server.
   * @param host proxy server host.
   * @param port proxy server port.
   * @param key proxy server key.
   */
  synchronized void connect(String host, int port, String key) {
    this.host = host;
    this.port = port;
    this.key = key;
    running = true;
    this.notify();
  }
}
