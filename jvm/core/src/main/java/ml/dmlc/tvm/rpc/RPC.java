package ml.dmlc.tvm.rpc;

import ml.dmlc.tvm.Function;

import java.util.HashMap;
import java.util.Map;

public class RPC {
  public static final int RPC_MAGIC = 0xff271;
  public static final int RPC_SESS_MASK = 128;

  private static ThreadLocal<Map<String, Function>> apiFuncs
      = new ThreadLocal<Map<String, Function>>() {
    @Override
    protected Map<String, Function> initialValue() {
      return new HashMap<String, Function>();
    }
  };

  static Function getApi(String name) {
    Function func = apiFuncs.get().get(name);
    if (func == null) {
      func = Function.getFunction("contrib.rpc." + name);
      if (func == null) {
        return null;
      }
      apiFuncs.get().put(name, func);
    }
    return func;
  }
}
