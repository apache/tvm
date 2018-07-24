package ml.dmlc.tvm;

import ml.dmlc.tvm.rpc.Server;

import java.io.IOException;

public class TestUtils {
  public static class RefInt {
    public int value;
  }

  public static Server startServer(RefInt portRef) {
    Server server = null;
    int port = 9981;
    for (int i = 0; i < 10; ++i) {
      try {
        server = new Server(port + i);
        server.start();
        portRef.value = port + i;
        return server;
      } catch (IOException e) {
      }
    }
    throw new RuntimeException("Cannot find an available port.");
  }
}
