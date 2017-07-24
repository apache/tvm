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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Server extends ServerSocket {
  private final ConnectionThread connectionThread;
  private SocketFileDescriptorGetter socketFdGetter = new SocketFileDescriptorGetter() {
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

  public Server(int serverPort) throws IOException {
    super(serverPort);
    connectionThread = new ConnectionThread(this, socketFdGetter);
  }

  public void start() {
    connectionThread.start();
  }

  public void terminate() {
    connectionThread.terminate();
  }

  public void registerFilDescriptorGetter(SocketFileDescriptorGetter getter) {
    socketFdGetter = getter;
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
      System.err.println("Socket fd = " + sockFd);
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
      });

      Function.register("tvm.contrib.rpc.server.load_module", new Function.Callback() {
        @Override public Object invoke(TVMValue... args) {
          String filename = args[0].asString();
          String path = tempDir + File.separator + filename;
          // Try create a shared library in remote.
          if (path.endsWith(".o")) {
            System.err.println("Create shared library based on " + path);
            // TODO(yizhi): create .so
          }
          System.err.println("Load module from " + path);
          return Module.load(path);
        }
      }, true);

      return tempDir;
    }
  }

  static class ConnectionThread extends Thread {
    private final ServerSocket server;
    private final SocketFileDescriptorGetter socketFileDescriptorGetter;
    private volatile boolean running = true;

    public ConnectionThread(Server server, SocketFileDescriptorGetter sockFdGetter)
        throws IOException {
      this.server = server;
      this.socketFileDescriptorGetter = sockFdGetter;
    }

    public void terminate() {
      this.running = false;
    }

    @Override public void run() {
      while (running) {
        Socket socket = null;
        try {
          socket = server.accept();
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

          // TODO(yizhi): use ExecutorService, i.e., ThreadPool
          Thread processThread = new Thread(new ServerLoop(socket, socketFileDescriptorGetter));
          processThread.setDaemon(true);
          processThread.start();
        } catch (IOException e) {
          e.printStackTrace();
          terminate();
        }
      }
    }

    private byte[] recvAll(final InputStream in, final int nBytes) throws IOException {
      byte[] res = new byte[nBytes];
      int nRead = 0;
      while (nRead < nBytes) {
        int chunk = in.read(res, nRead, Math.min(nBytes - nRead, 1024));
        nRead += chunk;
      }
      return res;
    }
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

  private static String decodeToStr(byte[] bytes) {
    StringBuilder builder = new StringBuilder();
    for (byte bt : bytes) {
      builder.append((char) bt);
    }
    return builder.toString();
  }
}
