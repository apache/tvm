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
  private ConnectProxyServerProcessor currProcessor;
  private final Handler uiHandler;

  static final SocketFileDescriptorGetter socketFdGetter
      = new SocketFileDescriptorGetter() {
        @Override
        public int get(Socket socket) {
          return ParcelFileDescriptor.fromSocket(socket).getFd();
        }
      };

  RPCProcessor(Handler uiHandler) {
    this.uiHandler = uiHandler;
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
        currProcessor = new ConnectProxyServerProcessor(host, port, key, socketFdGetter);
      }
      try {
        currProcessor.run();
      } catch (Throwable e) {
        disconnect();
        // turn connect switch off.
        Message message = new Message();
        message.what = MainActivity.MSG_RPC_ERROR;
        Bundle bundle = new Bundle();
        bundle.putString(MainActivity.MSG_RPC_ERROR_DATA_KEY, e.getMessage());
        message.setData(bundle);
        uiHandler.sendMessage(message);
      }
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
    notify();
  }
}
