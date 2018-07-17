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

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.ParcelFileDescriptor;

import java.net.Socket;

import ml.dmlc.tvm.rpc.ConnectProxyServerProcessor;
import ml.dmlc.tvm.rpc.SocketFileDescriptorGetter;

/**
 * Connect to RPC proxy and deal with requests.
 */
class RPCProcessor extends Thread {
  private String host;
  private int port;
  private String key;
  private boolean running = false;
  private long startTime;
  private ConnectProxyServerProcessor currProcessor;
  private boolean kill = false;
  public static final int SESSION_TIMEOUT = 30000;

  static final SocketFileDescriptorGetter socketFdGetter
      = new SocketFileDescriptorGetter() {
        @Override
        public int get(Socket socket) {
          return ParcelFileDescriptor.fromSocket(socket).getFd();
        }
      };
  // callback to initialize the start time of an rpc session
  class setTimeCallback implements Runnable { 
    private RPCProcessor rPCProcessor;

    public setTimeCallback(RPCProcessor rPCProcessor) {
        this.rPCProcessor = rPCProcessor;    
    }

    @Override
    public void run() {
        rPCProcessor.setStartTime();
    }
  }

  @Override public void run() {
    while (true) {
      synchronized (this) {
        currProcessor = null;
        while (!running) {
          try {
            this.wait();
          } catch (InterruptedException e) {
          }
        }
        // if kill, we do nothing and wait for app restart
        // to prevent race where timedOut was reported but restart has not
        // happened yet
        if (kill) {
            System.err.println("waiting for restart...");
            currProcessor = null;
        }
        else {
            startTime = 0;
            currProcessor = new ConnectProxyServerProcessor(host, port, key, socketFdGetter);
            currProcessor.setStartTimeCallback(new setTimeCallback(this));
        }
      }
        if (currProcessor != null)
            currProcessor.run();
    }
  }

  /**
   * check if the current RPCProcessor has timed out while in a session
   */
  synchronized boolean timedOut(long curTime) {
    if (startTime == 0) {
        return false;
    }
    else if ((curTime - startTime) > SESSION_TIMEOUT) {
        System.err.println("set kill flag...");
        kill = true;
        return true;
    }
    return false;
  }

  /**
   * set the start time of the current RPC session (used in callback)
   */
  synchronized void setStartTime() {
    startTime = System.currentTimeMillis();
    System.err.println("start time set to: " + startTime);
  }

  synchronized long getStartTime() {
    return startTime;
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
    notify();
  }
}
