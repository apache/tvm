package org.apache.tvm.rpc;

import org.apache.tvm.Function;
import org.apache.tvm.TVMValue;
import org.apache.tvm.TVMValueBytes;

import java.io.IOException;
import java.net.Socket;

public class SocketChannel {
  private final Socket socket;

  SocketChannel(Socket sock) {
    socket = sock;
  }

  private Function fsend = Function.convertFunc(new Function.Callback() {
    @Override public Object invoke(TVMValue... args) {
      byte[] data = args[0].asBytes();
      try {
        socket.getOutputStream().write(data);
      } catch (IOException e) {
        e.printStackTrace();
        return -1;
      }
      return data.length;
    }
  });

  private Function frecv = Function.convertFunc(new Function.Callback() {
    @Override public Object invoke(TVMValue... args) {
      long size = args[0].asLong();
      try {
        return new TVMValueBytes(Utils.recvAll(socket.getInputStream(), (int) size));
      } catch (IOException e) {
        e.printStackTrace();
        return -1;
      }
    }
  });

  public Function getFsend() {
    return fsend;
  }

  public Function getFrecv() {
    return frecv;
  }
}
