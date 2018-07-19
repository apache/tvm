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
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.util.List;


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
  private int serverPort = 5001;
  public static final int MAX_SERVER_PORT = 5555;
  // time to wait before aborting tracker connection (ms)
  public static final int TRACKER_TIMEOUT = 6000;
  // time to wait for a connection before refreshing tracker connection (ms)
  public static final int STALE_TRACKER_TIMEOUT = 300000;
  // time to wait if no timeout value is specified (seconds)
  public static final int HARD_TIMEOUT_DEFAULT = 300;
  private RPCWatchdog watchdog;

  public ConnectTrackerServerProcessor(String trackerHost, int trackerPort, String key,
      SocketFileDescriptorGetter sockFdGetter, RPCWatchdog watchdog) throws IOException {
    while (true) {
        try {
            this.server = new ServerSocket(serverPort);
            server.setSoTimeout(STALE_TRACKER_TIMEOUT);
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
    this.watchdog = watchdog;
  }

  public String getMatchKey() {
    return matchKey;
  }

  @Override public void terminate() {
    try {
      server.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override public void run() {
    Socket trackerSocket = null;
    String recvKey = null;
    try {
      trackerSocket = connectToTracker();
      register(trackerSocket);
      Socket socket = null;
      InputStream in = null;
      OutputStream out = null;
      while (true) {
        try {
          System.err.println("waiting for requests...");
          socket = server.accept();
        in = socket.getInputStream();
        out = socket.getOutputStream();
        int magic = Utils.wrapBytes(Utils.recvAll(in, 4)).getInt();
        if (magic != RPC.RPC_MAGIC) {
          out.write(Utils.toBytes(RPC.RPC_CODE_MISMATCH));
          System.err.println("incorrect RPC magic");
          Utils.closeQuietly(socket);
          continue;
        }
        int keyLen = Utils.wrapBytes(Utils.recvAll(in, 4)).getInt();
        recvKey = Utils.decodeToStr(Utils.recvAll(in, keyLen));
        System.err.println("matchKey:" + matchKey);
        System.err.println("key: " + recvKey);
        // incorrect key
        if (recvKey.indexOf(matchKey) == -1) {
          out.write(Utils.toBytes(RPC.RPC_CODE_MISMATCH));
          System.err.println("key mismatch, expected: " + matchKey + " got: " +  recvKey);
          Utils.closeQuietly(socket);
          continue;
        }
        break;
        } catch (SocketTimeoutException e) {
          System.err.println("no incoming connections, refreshing...");
          // need to reregister, if the tracker died we should see a socked closed exception
          if (!checkMatchKey(trackerSocket)) {
            System.err.println("reregistering...");
            register(trackerSocket);
          }
        }
      }
      int timeout = HARD_TIMEOUT_DEFAULT;
      int timeoutArgIndex = recvKey.indexOf(RPC.TIMEOUT_ARG);
      if (timeoutArgIndex != -1) {
        timeout = Integer.parseInt(recvKey.substring(timeoutArgIndex + RPC.TIMEOUT_ARG.length()));
      }
      System.err.println("alloted timeout: " + timeout);
      if (!recvKey.startsWith("client:")) {
        System.err.println("recv key mismatch...");
        out.write(Utils.toBytes(RPC.RPC_CODE_MISMATCH));
      }
      else {
        out.write(Utils.toBytes(RPC.RPC_MAGIC));
        // send server key to the client
        out.write(Utils.toBytes(recvKey.length()));
        out.write(Utils.toBytes(recvKey));
      }

      System.err.println("Connection from " + socket.getRemoteSocketAddress().toString());
      // received timeout in seconds
      watchdog.startTimeout(timeout*1000);
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
            if (trackerSocket != null) {
                trackerSocket.close();
            }
            server.close();
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }
  }

  private Socket connectToTracker() throws IOException{
    Socket trackerSocket = new Socket();
    SocketAddress address = new InetSocketAddress(trackerHost, trackerPort);
    trackerSocket.connect(address, TRACKER_TIMEOUT);
    InputStream trackerIn = trackerSocket.getInputStream();
    OutputStream trackerOut = trackerSocket.getOutputStream();
    trackerOut.write(Utils.toBytes(RPC.RPC_TRACKER_MAGIC));
    int trackerMagic = Utils.wrapBytes(Utils.recvAll(trackerIn, 4)).getInt();
    if (trackerMagic != RPC.RPC_TRACKER_MAGIC) {
      throw new SocketException("failed to connect to tracker (WRONG MAGIC)");
    }
    return trackerSocket;
  }

  private void register(Socket trackerSocket) throws IOException {
     InputStream trackerIn = trackerSocket.getInputStream();
     OutputStream trackerOut = trackerSocket.getOutputStream();
     String putJSON = generatePut(RPC.TrackerCode.PUT, key, serverPort, matchKey);
     trackerOut.write(Utils.toBytes(putJSON.length()));
     trackerOut.write(Utils.toBytes(putJSON));
     int recvLen = Utils.wrapBytes(Utils.recvAll(trackerIn, 4)).getInt();
     int recvCode = Integer.parseInt(Utils.decodeToStr(Utils.recvAll(trackerIn, recvLen)));
     if (recvCode != RPC.TrackerCode.SUCCESS) {
       throw new SocketException("failed to register with tracker (not SUCCESS)");
     }
     System.err.println("registered with tracker...");
  }

  // if we find the matchKey, we do not need to refresh
  private boolean checkMatchKey(Socket trackerSocket) throws IOException {
    InputStream trackerIn = trackerSocket.getInputStream();
    OutputStream trackerOut = trackerSocket.getOutputStream();
    String getJSON = generateGetPendingMatchKeys(RPC.TrackerCode.GET_PENDING_MATCHKEYS);
    trackerOut.write(Utils.toBytes(getJSON.length()));
    trackerOut.write(Utils.toBytes(getJSON));
    int recvLen = Utils.wrapBytes(Utils.recvAll(trackerIn, 4)).getInt();
    String recvJSON = Utils.decodeToStr(Utils.recvAll(trackerIn, recvLen));
    System.err.println("pending matchkeys: " + recvJSON);
    // fairly expensive string operation...
    if (recvJSON.indexOf(matchKey) != -1 ) {
      return true;
    }
    return false;
  }

  // handcrafted JSON
  private String generatePut(int code, String key, int port, String matchKey) {
    return "[" + code + ", " + "\"" + key + "\"" + ", " + "[" + port + ", " +
                      "\"" + matchKey +  "\"" + "]" + ", " + "null" + "]";
  }

  // handcrafted JSON
  private String generateGetPendingMatchKeys(int code) {
    return "[" + code  + "]";
  }
}
