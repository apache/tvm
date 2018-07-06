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

package ml.dmlc.tvm.tvmrpc;

import android.content.Context;
import android.content.Intent;

/**
 * Watchdog for RPCService
 */
class RPCWatchdog extends Thread {
  public static final int WATCHDOG_POLL_INTERVAL = 5000;
  private String host;
  private int port;
  private String key;
  private Context context;
  private boolean done = false;

  public RPCWatchdog(String host, int port, String key, Context context) {
    super();
    this.host = host;
    this.port = port;
    this.key = key;
    this.context = context;
  }

  /**
   * Polling loop to check on RPCService status
   */ 
  @Override public void run() {
    try {
        while (true) {
          synchronized (this) {
              if (done) {
                System.err.println("watchdog done, returning...");
                return;
              }
              else {
                System.err.println("polling rpc service...");                                  
                System.err.println("sending rpc service intent...");
                Intent intent = new Intent(context, RPCService.class);
                intent.putExtra("host", host);
                intent.putExtra("port", port);
                intent.putExtra("key", key);
                // will implicilty restart the service if it died
                context.startService(intent);
              }
          }
          Thread.sleep(WATCHDOG_POLL_INTERVAL);
        }
    } catch (InterruptedException e) {
    }
  }

  /**
   * Disconnect from the proxy server.
   */
  synchronized void disconnect() {
    // kill service
    System.err.println("watchdog disconnect call...");
    System.err.println("stopping rpc service...");
    done = true;
    Intent intent = new Intent(context, RPCService.class);
    intent.putExtra("kill", true); 
    context.startService(intent); 
  }
}
