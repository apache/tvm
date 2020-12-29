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

/**
 * Watchdog for RPC.
 */
public class RPCWatchdog extends Thread {
  private int timeout = 0;
  private boolean started = false;

  public RPCWatchdog() {
    super();
  }

  /**
   * Start a timeout with watchdog (must be called before finishTimeout).
   * @param timeout watchdog timeout in ms.
   */
  public synchronized void startTimeout(int timeout) {
    this.timeout = timeout;
    started = true;
    this.notify();
  }

  /**
   * Finish a timeout with watchdog (must be called after startTimeout).
   */
  public synchronized void finishTimeout() {
    started = false;
    this.notify();
  }

  /**
   * Wait and kill RPC if timeout is exceeded.
   */
  @Override public void run() {
    while (true) {
      // timeout not started
      synchronized (this) {
        while (!started) {
          try {
            this.wait();
          } catch (InterruptedException e) {
            System.err.println("watchdog interrupted...");
          }
        }
      }
      synchronized (this) {
        while (started) {
          try {
            System.err.println("waiting for timeout: " + timeout);
            this.wait(timeout);
            if (!started) {
              System.err.println("watchdog woken up, ok...");
            } else {
              System.err.println("watchdog woke up!");
              System.err.println("terminating...");
              terminate();
            }
          } catch (InterruptedException e) {
            System.err.println("watchdog interrupted...");
          }
        }
      }
    }
  }

  /**
   * Default method to terminate the running RPCActivity process.
   */
  protected void terminate() {
    System.exit(0);
  }
}
