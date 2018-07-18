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
import java.net.BindException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.Random;


/**
 * Server processor with tracker connection (based on standalone).
 */
public class ConnectTrackerServerProcessor implements ServerProcessor {
  private ServerSocket server;
  private final SocketFileDescriptorGetter socketFileDescriptorGetter;
  private final String trackerHost;
  private final int trackerPort;
  private final String key;
  private final String matchKey;
  Runnable callback;
  private int serverPort = 5001;
  public static final int MAX_SERVER_PORT = 5555;
  public static final int TRACKER_TIMEOUT = 6000;

  public ConnectTrackerServerProcessor(String trackerHost, int trackerPort, String key,
      SocketFileDescriptorGetter sockFdGetter) throws IOException {
    while (true) { 
        try {
            this.server = new ServerSocket(serverPort);
            break;
        } catch (BindException e) {
            System.err.println(serverPort);
            System.err.println(e);
            serverPort++; 
            if (serverPort > MAX_SERVER_PORT) {
                throw e;
            }
        }
    }
    System.err.println("using port: " + serverPort);
    this.socketFileDescriptorGetter = sockFdGetter;
    this.trackerHost = trackerHost;
    this.trackerPort = trackerPort;
    this.key = key; 
    this.matchKey = key + ":" + Math.random();
  }

  public String getMatchKey() {
    return matchKey;
  }

  /** 
   * Set a callback when a connection is received e.g., to record the time for a
   * watchdog.
   * @param callback Runnable object.
   */
  public void setStartTimeCallback(Runnable callback) {
    this.callback = callback;
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
      Socket trackerSocket = new Socket();
      SocketAddress address = new InetSocketAddress(trackerHost, trackerPort);
      trackerSocket.connect(address, 6000);
      InputStream trackerIn = trackerSocket.getInputStream();
      OutputStream trackerOut = trackerSocket.getOutputStream();
      String putJSON = generatePut(RPC.TrackerCode.PUT, "servertest1337", serverPort, matchKey);
      System.err.println(putJSON);
      trackerOut.write(Utils.toBytes(RPC.RPC_TRACKER_MAGIC));
      trackerOut.write(Utils.toBytes(putJSON.length())); 
      trackerOut.write(Utils.toBytes(putJSON));
      int trackerMagic = Utils.wrapBytes(Utils.recvAll(trackerIn, 4)).getInt();
      if (trackerMagic != RPC.RPC_TRACKER_MAGIC) {
        Utils.closeQuietly(trackerSocket);
        return;
      }

      int recvLen = Utils.wrapBytes(Utils.recvAll(trackerIn, 4)).getInt();
      int recvCode = Integer.parseInt(Utils.decodeToStr(Utils.recvAll(trackerIn, recvLen)));
      if (recvCode != RPC.TrackerCode.SUCCESS) {
        Utils.closeQuietly(trackerSocket);
        return;
      }
      trackerSocket.close();   
      System.err.println("registered with tracker...");

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
      System.err.println("matchKey:" + matchKey);
      System.err.println("key: " + key);
        
      int timeout = 0;
      int timeoutArgIndex = key.indexOf(RPC.TIMEOUT_ARG);
      if (timeoutArgIndex != -1) {
        timeout = Integer.parseInt(key.substring(timeoutArgIndex + RPC.TIMEOUT_ARG.length()));
      }
      System.err.println("alloted timeout: " + timeout);
      if (!key.startsWith("client:")) {
        out.write(Utils.toBytes(RPC.RPC_MAGIC + 2));
      } else {
        out.write(Utils.toBytes(RPC.RPC_MAGIC));
        // send server key to the client
        String serverKey = "server:java";
        out.write(Utils.toBytes(serverKey.length()));
        out.write(Utils.toBytes(serverKey));
      }

      System.err.println("Connection from " + socket.getRemoteSocketAddress().toString());
      if (callback != null) {
        callback.run();
      }

      final int sockFd = socketFileDescriptorGetter.get(socket);
      if (sockFd != -1) {
        new NativeServerLoop(sockFd).run();
        System.err.println("Finish serving " + socket.getRemoteSocketAddress().toString());
      }
      Utils.closeQuietly(socket);
    } catch (Throwable e) {
      e.printStackTrace();
    } finally {
        try {
            server.close();
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }
  }

  // handcrafted JSON
  private String generatePut(int code, String key, int port, String matchKey) {
    return "[" + code + ", " + "\"" + key + "\"" + ", " + "[" + port + ", " +
                      "\"" + matchKey +  "\"" + "]" + ", " + "null" + "]";
  }
}
