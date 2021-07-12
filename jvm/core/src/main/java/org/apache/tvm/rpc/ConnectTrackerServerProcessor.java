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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.BindException;
import java.net.ConnectException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketAddress;
import java.net.SocketException;
import java.net.SocketTimeoutException;


/**
 * Server processor with tracker connection (based on standalone).
 * This RPC Server registers itself with an RPC Tracker for a specific queue
 * (using its device key) and listens for incoming requests.
 */
public class ConnectTrackerServerProcessor implements ServerProcessor {
  private ServerSocket server;
  private final String trackerHost;
  private final int trackerPort;
  // device key
  private final String key;
  // device key plus randomly generated key (per-session)
  private final String matchKey;
  private int serverPort = 5001;
  public static final int MAX_SERVER_PORT = 5555;
  // time to wait before aborting tracker connection (ms)
  public static final int TRACKER_TIMEOUT = 6000;
  // time to wait before retrying tracker connection (ms)
  public static final int RETRY_PERIOD = TRACKER_TIMEOUT;
  // time to wait for a connection before refreshing tracker connection (ms)
  public static final int STALE_TRACKER_TIMEOUT = 300000;
  // time to wait if no timeout value is specified (seconds)
  public static final int HARD_TIMEOUT_DEFAULT = 300;
  private RPCWatchdog watchdog;
  private Socket trackerSocket;

  /**
   * Construct tracker server processor.
   * @param trackerHost Tracker host.
   * @param trackerPort Tracker port.
   * @param key Device key.
   * @param watchdog watch for timeout, etc.
   * @throws java.io.IOException when socket fails to open.
   */
  public ConnectTrackerServerProcessor(String trackerHost, int trackerPort, String key,
      RPCWatchdog watchdog) throws IOException {
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
    String recvKey = null;
    try {
      trackerSocket = connectToTracker();
      // open a socket and handshake with tracker
      register();
      Socket socket = null;
      InputStream in = null;
      OutputStream out = null;
      while (true) {
        try {
          System.err.println("waiting for requests...");
          // wait for client request
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
          recvKey = Utils.recvString(in);
          System.err.println("matchKey:" + matchKey);
          System.err.println("key: " + recvKey);
          // incorrect key
          if (recvKey.indexOf(matchKey) == -1) {
            out.write(Utils.toBytes(RPC.RPC_CODE_MISMATCH));
            System.err.println("key mismatch, expected: " + matchKey + " got: " +  recvKey);
            Utils.closeQuietly(socket);
            continue;
          }
          // successfully got client request and completed handshake with client
          break;
        } catch (SocketTimeoutException e) {
          System.err.println("no incoming connections, refreshing...");
          // need to reregister, if the tracker died we should see a socked closed exception
          if (!needRefreshKey()) {
            System.err.println("reregistering...");
            register();
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
      } else {
        out.write(Utils.toBytes(RPC.RPC_MAGIC));
        // send server key to the client
        Utils.sendString(out, recvKey);
      }

      System.err.println("Connection from " + socket.getRemoteSocketAddress().toString());
      // received timeout in seconds
      watchdog.startTimeout(timeout * 1000);
      SocketChannel sockChannel = new SocketChannel(socket);
      new NativeServerLoop(sockChannel.getFsend(), sockChannel.getFrecv()).run();
      System.err.println("Finish serving " + socket.getRemoteSocketAddress().toString());
      Utils.closeQuietly(socket);
    } catch (ConnectException e) {
      // if the tracker connection failed, wait a bit before retrying
      try {
        Thread.sleep(RETRY_PERIOD);
      } catch (InterruptedException e_) {
        System.err.println("interrupted before retrying to connect to tracker...");
      }
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

  private Socket connectToTracker() throws IOException {
    trackerSocket = new Socket();
    SocketAddress address = new InetSocketAddress(trackerHost, trackerPort);
    trackerSocket.connect(address, TRACKER_TIMEOUT);
    InputStream trackerIn = trackerSocket.getInputStream();
    OutputStream trackerOut = trackerSocket.getOutputStream();
    trackerOut.write(Utils.toBytes(RPC.RPC_TRACKER_MAGIC));
    int trackerMagic = Utils.wrapBytes(Utils.recvAll(trackerIn, 4)).getInt();
    if (trackerMagic != RPC.RPC_TRACKER_MAGIC) {
      throw new SocketException("failed to connect to tracker (WRONG MAGIC)");
    }
    String infoJSON = generateCinfo(key);
    Utils.sendString(trackerOut, infoJSON);
    int recvCode = Integer.parseInt(Utils.recvString(trackerIn));
    if (recvCode != RPC.TrackerCode.SUCCESS) {
      throw new SocketException("failed to connect to tracker (not SUCCESS)");
    }
    return trackerSocket;
  }

  /*
   * Register the RPC Server with the RPC Tracker.
   */
  private void register() throws IOException {
    InputStream trackerIn = trackerSocket.getInputStream();
    OutputStream trackerOut = trackerSocket.getOutputStream();
    // send a JSON with PUT, device key, RPC server port, and the randomly
    // generated key
    String putJSON = generatePut(RPC.TrackerCode.PUT, key, serverPort, matchKey);
    Utils.sendString(trackerOut, putJSON);
    int recvCode = Integer.parseInt(Utils.recvString(trackerIn));
    if (recvCode != RPC.TrackerCode.SUCCESS) {
      throw new SocketException("failed to register with tracker (not SUCCESS)");
    }
    System.err.println("registered with tracker...");
  }

  /*
   * Check if the RPC Tracker has our key.
   */
  private boolean needRefreshKey() throws IOException {
    InputStream trackerIn = trackerSocket.getInputStream();
    OutputStream trackerOut = trackerSocket.getOutputStream();
    String getJSON = generateGetPendingMatchKeys(RPC.TrackerCode.GET_PENDING_MATCHKEYS);
    Utils.sendString(trackerOut, getJSON);
    String recvJSON = Utils.recvString(trackerIn);
    System.err.println("pending matchkeys: " + recvJSON);
    // fairly expensive string operation...
    if (recvJSON.indexOf(matchKey) != -1 ) {
      return true;
    }
    return false;
  }

  // handcrafted JSON
  private String generateCinfo(String key) {
    String cinfo = "{\"key\" : " + "\"server:" + key + "\", \"addr\": [null, \""
        + serverPort + "\"]}";
    return "[" + RPC.TrackerCode.UPDATE_INFO + ", " + cinfo + "]";
  }

  // handcrafted JSON
  private String generatePut(int code, String key, int port, String matchKey) {
    return "[" + code + ", " + "\"" + key + "\"" + ", " + "[" + port + ", "
            + "\"" + matchKey +  "\"" + "]" + ", " + "null" + "]";
  }

  // handcrafted JSON
  private String generateGetPendingMatchKeys(int code) {
    return "[" + code  + "]";
  }
}
