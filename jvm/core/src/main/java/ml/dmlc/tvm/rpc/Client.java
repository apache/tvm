package ml.dmlc.tvm.rpc;

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.TVMValue;

public class Client {
  /**
   * Connect to RPC Server.
   * @param url The url of the host.
   * @param port The port to connect to.
   * @param key Additional key to match server.
   * @return The connected session.
   */
  public static RPCSession connect(String url, int port, String key) {
    Function doConnect = RPC.getApi("_Connect");
    if (doConnect == null) {
      throw new RuntimeException("Please compile with USE_RPC=1");
    }
    TVMValue sess = doConnect.pushArg(url).pushArg(port).pushArg(key).invoke();
    return new RPCSession(sess.asModule());
  }

  /**
   * Connect to RPC Server.
   * @param url The url of the host.
   * @param port The port to connect to.
   * @return The connected session.
   */
  public static RPCSession connect(String url, int port) {
    return connect(url, port, "");
  }
}
