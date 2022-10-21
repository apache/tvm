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

import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;

/**
 * Server processor for proxy connection.
 */
public class ConnectProxyServerProcessor implements ServerProcessor {
  private final String host;
  private final int port;
  private final String key;

  private volatile Socket currSocket = new Socket();
  private Runnable callback;

  /**
   * Construct proxy server processor.
   * @param host Proxy server host.
   * @param port Proxy server port.
   * @param key Proxy server key.
   */
  public ConnectProxyServerProcessor(String host, int port, String key) {
    this.host = host;
    this.port = port;
    this.key = "server:" + key;
  }

  /**
   * Set a callback when a connection is received e.g., to record the time for a
   * watchdog.
   * @param callback Runnable object.
   */
  public void setStartTimeCallback(Runnable callback) {
    this.callback = callback;
  }

  /**
   * Close the socket.
   */
  @Override public void terminate() {
    Utils.closeQuietly(currSocket);
  }

  @Override public void run() {
    try {
      SocketAddress address = new InetSocketAddress(host, port);
      currSocket.connect(address, 6000);
      final InputStream in = currSocket.getInputStream();
      final OutputStream out = currSocket.getOutputStream();
      out.write(Utils.toBytes(RPC.RPC_MAGIC));
      out.write(Utils.toBytes(key.length()));
      out.write(Utils.toBytes(key));
      int magic = Utils.wrapBytes(Utils.recvAll(in, 4)).getInt();
      if (magic == RPC.RPC_MAGIC + 1) {
        throw new RuntimeException(
            String.format("key: %s has already been used in proxy", key));
      } else if (magic == RPC.RPC_MAGIC + 2) {
        System.err.println("RPCProxy do not have matching client key " + key);
      } else if (magic != RPC.RPC_MAGIC) {
        throw new RuntimeException(address + " is not RPC Proxy");
      }
      // Get key from remote
      int keylen = Utils.wrapBytes(Utils.recvAll(in, 4)).getInt();
      String remoteKey = Utils.decodeToStr(Utils.recvAll(in, keylen));
      System.err.println("RPCProxy connected to " + address);
      if (callback != null) {
        callback.run();
      }

      SocketChannel sockChannel = new SocketChannel(currSocket);
      new NativeServerLoop(sockChannel.getFsend(), sockChannel.getFrecv()).run();
      System.err.println("Finish serving " + address);
    } catch (Throwable e) {
      e.printStackTrace();
      throw new RuntimeException(e);
    } finally {
      terminate();
    }
  }
}
