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

import java.util.List;
import android.content.Context;
import android.content.Intent;
import android.app.ActivityManager;
//import android.app.Activity;

/**
 * Watchdog for RPCService
 */
class RPCWatchdog extends Thread {
  public static final int WATCHDOG_POLL_INTERVAL = 5000;
  //private String host;
  //private int port;
  //private String key;
  //private boolean done = false;
  private RPCProcessor tvmServerWorker;
  //private Context context;

  public RPCWatchdog(RPCProcessor tvmServerWorker) {
    super();
    this.tvmServerWorker = tvmServerWorker;
  }

  /**
   * Polling loop to check on RPCService status
   */ 
  @Override public void run() {
    try {
        while (true) {
          if (tvmServerWorker.timedOut(System.currentTimeMillis())) {
            System.err.println("rpc processor timed out, killing self...");  Thread.sleep(WATCHDOG_POLL_INTERVAL);
            System.exit(0);
          } else {
            System.err.println("rpc processor ok...");
          }
          Thread.sleep(WATCHDOG_POLL_INTERVAL);
        }
    } catch (InterruptedException e) {
    }
  }
}
