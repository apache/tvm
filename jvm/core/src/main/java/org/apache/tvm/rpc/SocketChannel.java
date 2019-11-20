/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
