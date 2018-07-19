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

/**
 * Watchdog for RPC
 */
public class RPCWatchdog extends Thread {
  int timeout = 0;
  private Thread tvmServerWorker;

  public RPCWatchdog(Thread tvmServerWorker) {
    super();
    this.tvmServerWorker = tvmServerWorker;
  }

  synchronized public void setTimeout(int timeout) {
    this.timeout = timeout;
  }

  /**
   * Wait and kill RPC if timeout is exceeded
   */ 
  @Override public void run() {
    try {
      System.err.println("starting watchdog...");
      synchronized(this) {
        this.wait(timeout);
      }
      System.err.println("watchdog woke up!");
      System.err.println("watchdog killing process...");
      // Prevent recycling of current process
      System.exit(0);
    } catch (InterruptedException e) {
      System.err.println("watchdog woken up, ok...");
      tvmServerWorker.notify();
    }
  }
}
