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

package ml.dmlc.tvm.rpc;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * Server processor for standalone.
 */
public class StandaloneServerProcessor implements ServerProcessor {
  private final ServerSocket server;
  private final SocketFileDescriptorGetter socketFileDescriptorGetter;

  public StandaloneServerProcessor(int serverPort,
      SocketFileDescriptorGetter sockFdGetter) throws IOException {
    this.server = new ServerSocket(serverPort);
    this.socketFileDescriptorGetter = sockFdGetter;
  }

  @Override public void terminate() {
    try {
      server.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override public void run() {
    try {
      Socket socket = server.accept();
      InputStream in = socket.getInputStream();
      OutputStream out = socket.getOutputStream();
      int magic = Utils.wrapBytes(Utils.recvAll(in, 4)).getInt();
      if (magic != RPC.RPC_MAGIC) {
        Utils.closeQuietly(socket);
        return;
      }
      int keyLen = Utils.wrapBytes(Utils.recvAll(in, 4)).getInt();
      String key = Utils.decodeToStr(Utils.recvAll(in, keyLen));
      if (!key.startsWith("client:")) {
        out.write(Utils.toBytes(RPC.RPC_MAGIC + 2));
      } else {
        out.write(Utils.toBytes(RPC.RPC_MAGIC));
      }
      System.err.println("Connection from " + socket.getRemoteSocketAddress().toString());
      final int sockFd = socketFileDescriptorGetter.get(socket);
      if (sockFd != -1) {
        new NativeServerLoop(sockFd).run();
        System.err.println("Finish serving " + socket.getRemoteSocketAddress().toString());
      }
      Utils.closeQuietly(socket);
    } catch (Throwable e) {
      e.printStackTrace();
    }
  }
}
