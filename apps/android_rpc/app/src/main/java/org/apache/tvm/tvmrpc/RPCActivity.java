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

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.widget.Button;
import android.view.View;

public class RPCActivity extends AppCompatActivity {
  private RPCProcessor tvmServerWorker;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_rpc);

    Button stopRPC = findViewById(R.id.button_stop_rpc);
    stopRPC.setOnClickListener(new View.OnClickListener() {
        public void onClick(View v) {
            System.err.println(tvmServerWorker == null);
            if (tvmServerWorker != null) {
                // currently will raise a socket closed exception
                tvmServerWorker.disconnect();
            }
            finish();
            // prevent Android from recycling the process
            System.exit(0);
        }
    });

    System.err.println("rpc activity onCreate...");
    Intent intent = getIntent();
    String host = intent.getStringExtra("host");
    int port = intent.getIntExtra("port", 9090);
    String key = intent.getStringExtra("key");

    tvmServerWorker = new RPCProcessor(this);
    tvmServerWorker.setDaemon(true);
    tvmServerWorker.start();
    tvmServerWorker.connect(host, port, key);
  }

  @Override
  protected void onDestroy() {
    System.err.println("rpc activity onDestroy");
    tvmServerWorker.disconnect();
    super.onDestroy();
    android.os.Process.killProcess(android.os.Process.myPid());
  }
}
