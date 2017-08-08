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

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.TVMValue;
import sun.misc.SharedSecrets;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * RPC Server.
 */
public class Server {
  private static SocketFileDescriptorGetter defaultSocketFdGetter
      = new SocketFileDescriptorGetter() {
          @Override public int get(Socket socket) {
            try {
              InputStream is = socket.getInputStream();
              FileDescriptor fd = ((FileInputStream) is).getFD();
              return SharedSecrets.getJavaIOFileDescriptorAccess().get(fd);
            } catch (IOException e) {
              e.printStackTrace();
              return -1;
            }
          }
        };
  private static final int DEFAULT_THREAD_NUMBER_IN_A_POOL = 20;

  private final Loop serverLoop;
  private final ExecutorService threadPool;

  /**
   * Start a standalone server.
   * @param serverPort Port.
   * @param socketFdGetter Method to get system file descriptor of the server socket.
   * @throws IOException if failed to bind localhost:port.
   */
  public Server(int serverPort, SocketFileDescriptorGetter socketFdGetter) throws IOException {
    threadPool = setupThreadPool();
    serverLoop = new ListenLoop(serverPort, threadPool, socketFdGetter);
  }

  /**
   * Start a standalone server.
   * Use sun.misc.SharedSecrets.getJavaIOFileDescriptorAccess
   * to get file descriptor for the socket.
   * @param serverPort Port.
   * @throws IOException if failed to bind localhost:port.
   */
  public Server(int serverPort) throws IOException {
    this(serverPort, defaultSocketFdGetter);
  }

  /**
   * Start a server connected to proxy.
   * @param proxyHost The proxy server host.
   * @param proxyPort The proxy server port.
   * @param key The key to identify the server.
   * @param socketFdGetter Method to get system file descriptor of the server socket.
   */
  public Server(String proxyHost, int proxyPort, String key,
      SocketFileDescriptorGetter socketFdGetter) {
    threadPool = setupThreadPool();
    serverLoop = new ConnectProxyLoop(proxyHost, proxyPort, key, threadPool, socketFdGetter);
  }

  /**
   * Start a server connected to proxy.
   * Use sun.misc.SharedSecrets.getJavaIOFileDescriptorAccess
   * to get file descriptor for the socket.
   * @param proxyHost The proxy server host.
   * @param proxyPort The proxy server port.
   * @param key The key to identify the server.
   */
  public Server(String proxyHost, int proxyPort, String key) {
    this(proxyHost, proxyPort, key, defaultSocketFdGetter);
  }

  private ExecutorService setupThreadPool() {
    final String workerThreadNumber = System.getProperty("rpc.server.thread.number");
    final int numThread = (workerThreadNumber == null)
        ? DEFAULT_THREAD_NUMBER_IN_A_POOL : Integer.parseInt(workerThreadNumber);
    return Executors.newFixedThreadPool(numThread);
  }

  /**
   * Start the server.
   */
  public void start() {
    serverLoop.start();
  }

  /**
   * Stop the server.
   */
  public void terminate() {
    serverLoop.interrupt();
    serverLoop.terminate();
    threadPool.shutdown();
  }

  public static interface SocketFileDescriptorGetter {
    public int get(Socket socket);
  }

  static class ServerLoop implements Runnable {
    private final Socket socket;
    private final SocketFileDescriptorGetter socketFdGetter;

    ServerLoop(Socket socket, SocketFileDescriptorGetter fdGetter) {
      this.socket = socket;
      socketFdGetter = fdGetter;
    }

    @Override public void run() {
      int sockFd = socketFdGetter.get(socket);
      if (sockFd != -1) {
        File tempDir = null;
        try {
          tempDir = serverEnv();
          RPC.getApi("_ServerLoop").pushArg(sockFd).invoke();
          System.err.println("Finish serving " + socket.getRemoteSocketAddress().toString());
        } catch (IOException e) {
          e.printStackTrace();
        } finally {
          if (tempDir != null) {
            if (!tempDir.delete()) {
              System.err.println(
                  "[WARN] Couldn't delete temporary directory " + tempDir.getAbsolutePath());
            }
          }
          closeQuietly(socket);
        }
      }
    }

    private File serverEnv() throws IOException {
      // Server environment function return temp dir.
      final File tempDir = File.createTempFile("tvm4j_rpc_", "");
      if (!tempDir.delete() || !tempDir.mkdir()) {
        throw new IOException("Couldn't create directory " + tempDir.getAbsolutePath());
      }

      Function.register("tvm.contrib.rpc.server.workpath", new Function.Callback() {
        @Override public Object invoke(TVMValue... args) {
          return tempDir + File.separator + args[0].asString();
        }
      }, true);

      Function.register("tvm.contrib.rpc.server.load_module", new Function.Callback() {
        @Override public Object invoke(TVMValue... args) {
          String filename = args[0].asString();
          String path = tempDir + File.separator + filename;
          System.err.println("Load module from " + path);
          return Module.load(path);
        }
      }, true);

      return tempDir;
    }
  }

  abstract static class Loop extends Thread {
    public abstract void terminate();
  }

  static class ConnectProxyLoop extends Loop {
    private volatile boolean running = true;
    private final String host;
    private final int port;
    private final String key;
    private final ExecutorService workerPool;
    private final SocketFileDescriptorGetter socketFileDescriptorGetter;
    private Socket waitingSocket = null;

    public ConnectProxyLoop(String host, int port, String key,
        ExecutorService workerPool,
        SocketFileDescriptorGetter sockFdGetter) {
      this.host = host;
      this.port = port;
      this.key = "server:" + key;
      this.workerPool = workerPool;
      socketFileDescriptorGetter = sockFdGetter;
    }

    @Override public void terminate() {
      running = false;
      if (waitingSocket != null) {
        try {
          waitingSocket.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }

    @Override public void run() {
      while (running) {
        try {
          Socket socket = new Socket(host, port);
          waitingSocket = socket;
          InputStream in = socket.getInputStream();
          OutputStream out = socket.getOutputStream();
          out.write(toBytes(RPC.RPC_MAGIC));
          out.write(toBytes(key.length()));
          out.write(toBytes(key));
          int magic = wrapBytes(recvAll(in, 4)).getInt();
          final String address = host + ":" + port;
          if (magic == RPC.RPC_MAGIC + 1) {
            throw new RuntimeException(
                String.format("key: %s has already been used in proxy", key));
          } else if (magic == RPC.RPC_MAGIC + 2) {
            System.err.println("RPCProxy do not have matching client key " + key);
          } else if (magic != RPC.RPC_MAGIC) {
            throw new RuntimeException(address + " is not RPC Proxy");
          }
          System.err.println("RPCProxy connected to " + address);

          waitingSocket = null;
          workerPool.execute(new ServerLoop(socket, socketFileDescriptorGetter));
        } catch (SocketException e) {
          // when terminates, this is what we expect, do nothing.
        } catch (IOException e) {
          e.printStackTrace();
          terminate();
        }
      }
    }
  }

  static class ListenLoop extends Loop {
    private final ServerSocket server;
    private final ExecutorService workerPool;
    private final SocketFileDescriptorGetter socketFileDescriptorGetter;
    private volatile boolean running = true;

    public ListenLoop(int serverPort, ExecutorService workerPool,
        SocketFileDescriptorGetter sockFdGetter) throws IOException {
      this.server = new ServerSocket(serverPort);
      this.workerPool = workerPool;
      this.socketFileDescriptorGetter = sockFdGetter;
    }

    @Override public void terminate() {
      this.running = false;
      try {
        server.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

    @Override public void run() {
      while (running) {
        try {
          Socket socket = server.accept();
          InputStream in = socket.getInputStream();
          OutputStream out = socket.getOutputStream();
          int magic = wrapBytes(recvAll(in, 4)).getInt();
          if (magic != RPC.RPC_MAGIC) {
            closeQuietly(socket);
            continue;
          }
          int keyLen = wrapBytes(recvAll(in, 4)).getInt();
          String key = decodeToStr(recvAll(in, keyLen));
          if (!key.startsWith("client:")) {
            out.write(toBytes(RPC.RPC_MAGIC + 2));
          } else {
            out.write(toBytes(RPC.RPC_MAGIC));
          }
          System.err.println("Connection from " + socket.getRemoteSocketAddress().toString());
          workerPool.execute(new ServerLoop(socket, socketFileDescriptorGetter));
        } catch (SocketException e) {
          // when terminates, this is what we expect, do nothing.
        } catch (IOException e) {
          e.printStackTrace();
          terminate();
        }
      }
    }
  }

  private static byte[] recvAll(final InputStream in, final int numBytes) throws IOException {
    byte[] res = new byte[numBytes];
    int numRead = 0;
    while (numRead < numBytes) {
      int chunk = in.read(res, numRead, Math.min(numBytes - numRead, 1024));
      numRead += chunk;
    }
    return res;
  }

  private static void closeQuietly(Socket socket) {
    if (socket != null) {
      try {
        socket.shutdownInput();
        socket.shutdownOutput();
        socket.close();
      } catch (IOException ioe) {
        // close quietly, do nothing.
      }
    }
  }

  private static ByteBuffer wrapBytes(byte[] bytes) {
    ByteBuffer bb = ByteBuffer.wrap(bytes);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    return bb;
  }

  private static byte[] toBytes(int number) {
    ByteBuffer bb = ByteBuffer.allocate(4);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    return bb.putInt(number).array();
  }

  private static byte[] toBytes(String str) {
    byte[] bytes = new byte[str.length()];
    for (int i = 0; i < str.length(); ++i) {
      bytes[i] = (byte) str.charAt(i);
    }
    return bytes;
  }

  private static String decodeToStr(byte[] bytes) {
    StringBuilder builder = new StringBuilder();
    for (byte bt : bytes) {
      builder.append((char) bt);
    }
    return builder.toString();
  }
}
