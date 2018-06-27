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

import android.app.ActivityManager;
import android.app.ActivityManager.RunningServiceInfo;
import android.content.Context;
import android.content.Intent;

/**
 * Watchdog for RPCService
 */
class RPCWatchdog extends Thread {
  private String m_host;
  private int m_port;
  private String m_key;
  private Context m_context;
  private boolean done = false;

  public RPCWatchdog(String host, int port, String key, Context context) {
    super();
    m_host = host;
    m_port = port;
    m_key = key;
    m_context = context;
  } 

  /**
   * Polling loop to check on RPCService status
   */ 

  @Override public void run() {
    try {
        while (true) {
          synchronized (this) {
              // check if rpc service exists
              // if rpc service is missing, restart
              boolean serviceRunning = rpcServiceRunning();
              if (done && serviceRunning) {
                System.err.println("watchdog done, returning...");
                Intent intent = new Intent(m_context, RPCService.class);
                m_context.stopService(intent);
                return;
              }
              else if (done && !serviceRunning) {
                System.err.println("watchdog done, returning...");
                return;
              }
              else if (!done && !serviceRunning) {
                System.err.println("creating rpc service...");
                System.err.println("sending rpc service intent...");
                Intent intent = new Intent(m_context, RPCService.class);
                intent.putExtra("host", m_host);
                intent.putExtra("port", m_port);
                intent.putExtra("key", m_key);
                m_context.startService(intent);
              }
          }
          Thread.sleep(5000);
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
  }

  public boolean rpcServiceRunning() {
    ActivityManager manager = (ActivityManager) m_context.getSystemService(Context.ACTIVITY_SERVICE);
    for (RunningServiceInfo service : manager.getRunningServices(Integer.MAX_VALUE)) {
        if (RPCService.class.getName().equals(service.service.getClassName())) {
            System.err.println("rpc service running...");
            return true;
        }
    }
    System.err.println("rpc service not running...");
    return false;
  }
}
