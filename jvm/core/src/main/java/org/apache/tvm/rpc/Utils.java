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
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Utilities for RPC.
 */
class Utils {
  public static byte[] recvAll(final InputStream in, final int numBytes) throws IOException {
    byte[] res = new byte[numBytes];
    int numRead = 0;
    while (numRead < numBytes) {
      int chunk = in.read(res, numRead, Math.min(numBytes - numRead, 1024));
      numRead += chunk;
    }
    return res;
  }

  public static void closeQuietly(Socket socket) {
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

  public static ByteBuffer wrapBytes(byte[] bytes) {
    ByteBuffer bb = ByteBuffer.wrap(bytes);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    return bb;
  }

  public static byte[] toBytes(int number) {
    ByteBuffer bb = ByteBuffer.allocate(4);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    return bb.putInt(number).array();
  }

  public static byte[] toBytes(String str) {
    byte[] bytes = new byte[str.length()];
    for (int i = 0; i < str.length(); ++i) {
      bytes[i] = (byte) str.charAt(i);
    }
    return bytes;
  }

  public static String decodeToStr(byte[] bytes) {
    StringBuilder builder = new StringBuilder();
    for (byte bt : bytes) {
      builder.append((char) bt);
    }
    return builder.toString();
  }

  public static String recvString(InputStream in) throws IOException {
    String recvString = null;
    int len = wrapBytes(Utils.recvAll(in, 4)).getInt();
    recvString = decodeToStr(Utils.recvAll(in, len));
    return recvString;
  }

  public static void sendString(OutputStream out, String string) throws IOException {
    out.write(toBytes(string.length()));
    out.write(toBytes(string));
  }
}
