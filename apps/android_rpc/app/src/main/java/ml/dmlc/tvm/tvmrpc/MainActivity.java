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

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;

public class MainActivity extends AppCompatActivity {
  static final int MSG_RPC_ERROR = 0;
  static final String MSG_RPC_ERROR_DATA_KEY = "msg_rpc_error_data_key";

  private RPCProcessor tvmServerWorker;
  @SuppressLint("HandlerLeak")
  private final Handler rpcHandler = new Handler() {
    @Override
    public void dispatchMessage(Message msg) {
      Switch switchConnect = findViewById(R.id.switch_connect);
      if (msg.what == MSG_RPC_ERROR && switchConnect.isChecked()) {
        // switch off and show alert dialog.
        switchConnect.setChecked(false);
        String msgBody = msg.getData().getString(MSG_RPC_ERROR_DATA_KEY);
        showDialog("Error", msgBody);
      }
    }
  };

  private void showDialog(String title, String msg) {
    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    builder.setTitle(title);
    builder.setMessage(msg);
    builder.setCancelable(true);
    builder.setNeutralButton(android.R.string.ok,
        new DialogInterface.OnClickListener() {
          public void onClick(DialogInterface dialog, int id) {
            dialog.cancel();
          }
        });
    builder.create().show();
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);

    tvmServerWorker = new RPCProcessor(rpcHandler);
    tvmServerWorker.setDaemon(true);
    tvmServerWorker.start();

    Switch switchConnect = findViewById(R.id.switch_connect);
    switchConnect.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if (isChecked) {
          enableInputView(false);
          connectProxy();
        } else {
          disconnect();
          enableInputView(true);
        }
      }
    });
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    tvmServerWorker.disconnect();
  }

  private void connectProxy() {
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);

    final String proxyHost = edProxyAddress.getText().toString();
    final int proxyPort = Integer.parseInt(edProxyPort.getText().toString());
    final String key = edAppKey.getText().toString();

    tvmServerWorker.connect(proxyHost, proxyPort, key);
  }

  private void disconnect() {
    tvmServerWorker.disconnect();
    System.err.println("Disconnected.");
  }

  private void enableInputView(boolean enable) {
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);
    edProxyAddress.setEnabled(enable);
    edProxyPort.setEnabled(enable);
    edAppKey.setEnabled(enable);
  }
}
