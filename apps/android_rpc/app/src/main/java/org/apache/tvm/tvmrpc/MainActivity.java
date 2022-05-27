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

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import android.widget.CompoundButton;
import android.widget.EditText;
import androidx.appcompat.widget.SwitchCompat;
import android.content.Intent;


public class MainActivity extends AppCompatActivity {
  // wait time before automatic restart of RPC Activity
  public static final int HANDLER_RESTART_DELAY = 5000;

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

  public Intent updateRPCPrefs() {
    System.err.println("updating preferences...");
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);
    SwitchCompat inputSwitch =  findViewById(R.id.switch_persistent);

    final String proxyHost = edProxyAddress.getText().toString();
    final int proxyPort = Integer.parseInt(edProxyPort.getText().toString());
    final String key = edAppKey.getText().toString();
    final boolean isChecked = inputSwitch.isChecked();

    SharedPreferences pref = getApplicationContext().getSharedPreferences("RPCProxyPreference", Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = pref.edit();
    editor.putString("input_address", proxyHost);
    editor.putString("input_port", edProxyPort.getText().toString());
    editor.putString("input_key", key);
    editor.putBoolean("input_switch", isChecked);
    editor.commit();

    Intent intent = new Intent(this, RPCActivity.class);
    intent.putExtra("host", proxyHost);
    intent.putExtra("port", proxyPort);
    intent.putExtra("key", key);
    return intent;
  }

  private void setupRelaunch() {
    final Context context = this;
    final SwitchCompat switchPersistent = findViewById(R.id.switch_persistent);
    final Runnable rPCStarter = new Runnable() {
        public void run() {
            if (switchPersistent.isChecked()) {
              System.err.println("relaunching RPC activity...");
              Intent intent = ((MainActivity) context).updateRPCPrefs();
              startActivity(intent);
            }
        }
    };

    Handler handler = new Handler(Looper.getMainLooper());
    handler.postDelayed(rPCStarter, HANDLER_RESTART_DELAY);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
    final Context context = this;

    SwitchCompat switchPersistent = findViewById(R.id.switch_persistent);
    switchPersistent.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if (isChecked) {
          System.err.println("automatic RPC restart enabled...");
          updateRPCPrefs();
          setupRelaunch();
        } else {
          System.err.println("automatic RPC restart disabled...");
          updateRPCPrefs();
        }
      }
    });

    enableInputView(true);
  }

  @Override
  protected void onResume() {
    System.err.println("MainActivity onResume...");
    enableInputView(true);
    setupRelaunch();
    super.onResume();
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
  }

  private void enableInputView(boolean enable) {
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);
    SwitchCompat input_switch = findViewById(R.id.switch_persistent);
    edProxyAddress.setEnabled(enable);
    edProxyPort.setEnabled(enable);
    edAppKey.setEnabled(enable);

    if (enable) {
    SharedPreferences pref = getApplicationContext().getSharedPreferences("RPCProxyPreference", Context.MODE_PRIVATE);
    String inputAddress = pref.getString("input_address", null);
    if (null != inputAddress)
        edProxyAddress.setText(inputAddress);
    String inputPort = pref.getString("input_port", null);
    if (null != inputPort)
        edProxyPort.setText(inputPort);
    String inputKey = pref.getString("input_key", null);
    if (null != inputKey)
        edAppKey.setText(inputKey);
    boolean isChecked = pref.getBoolean("input_switch", false);
    input_switch.setChecked(isChecked);
    }
  }
}
