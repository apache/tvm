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

package ml.dmlc.tvm.tvmrpc;

import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;

import java.net.Socket;

import ml.dmlc.tvm.rpc.Server;

public class MainActivity extends AppCompatActivity {
  static final Server.SocketFileDescriptorGetter socketFdGetter
      = new Server.SocketFileDescriptorGetter() {
        @Override
        public int get(Socket socket) {
          return ParcelFileDescriptor.fromSocket(socket).getFd();
        }
      };

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);

    Switch switchConnect = findViewById(R.id.switch_connect);
    switchConnect.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if (isChecked) {
          connectProxy();
        } else {
          disconnect();
        }
      }
    });
  }

  private Server tvmServer;

  private void connectProxy() {
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);

    final String proxyHost = edProxyAddress.getText().toString();
    final int proxyPort = Integer.parseInt(edProxyPort.getText().toString());
    final String key = edAppKey.getText().toString();

    if (tvmServer != null) {
      tvmServer.terminate();
    }

    tvmServer = new Server(proxyHost, proxyPort, key, socketFdGetter);
    tvmServer.start();
  }

  private void disconnect() {
    if (tvmServer != null) {
      tvmServer.terminate();
      System.err.println("Disconnected.");
    }
  }
}
