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

package org.apache.tvm.tvmrpc;

import android.app.Activity;
import org.apache.tvm.rpc.RPCWatchdog;

/**
 * Watchdog for Android RPC.
 */
public class RPCAndroidWatchdog extends RPCWatchdog {
  public Activity rpc_activity = null;
  public RPCAndroidWatchdog(Activity activity) {
    super();
    rpc_activity = activity;
  }

  /**
   * Method to non-destructively terminate the running thread on Android
   */
  @Override protected void terminate() {
    rpc_activity.finish();
  }
}
