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
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;

import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.Button;
import android.view.View;
import android.content.Intent;
import android.app.NotificationChannel;
import android.app.NotificationManager;


public class MainActivity extends AppCompatActivity {
  private int num;
 
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
    System.err.println("running update...");
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);
    Switch inputSwitch =  findViewById(R.id.switch_connect);
    
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
    System.err.println("done update...");
    return intent;
  }

  private void setupRelaunch() {
    final Context context = this;
    final Switch switchConnect = findViewById(R.id.switch_connect);
    final Runnable rPCStarter = new Runnable() {
        public void run() {
            if (switchConnect.isChecked()) {
              System.err.println("relaunching RPC activity in 5s...");
              Intent intent = ((MainActivity) context).updateRPCPrefs(); 
              startActivity(intent);
            }
        }
    };
    Handler handler = new Handler();
    handler.postDelayed(rPCStarter, 5000);
  }


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
    final Context context = this;
    
    Switch switchConnect = findViewById(R.id.switch_connect);
    switchConnect.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      @Override
      public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if (isChecked) {
          System.err.println("automatic RPC restart enabled...");
          updateRPCPrefs();
        } else {
          System.err.println("automatic RPC restart disabled...");
          updateRPCPrefs();
        }
      }
    }); 

    Button startRPC = findViewById(R.id.button_start_rpc);
    startRPC.setOnClickListener(new View.OnClickListener() {
        public void onClick(View v) {
            Intent intent = ((MainActivity) context).updateRPCPrefs(); 
            startActivity(intent);
        }
    });

    System.err.println("MainActivity onCreate...");
    System.err.println("num: " + num);
    num++;

    enableInputView(true);
    //setupRelaunch();
  }

  @Override
  protected void onResume() {
    System.err.println("MainActivity onResume...");
    System.err.println("num: " + num);
    num++;

    enableInputView(true);
    setupRelaunch();

    super.onResume();
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
  }

  private void disconnect() {
    // TODO disconnect standalone
  }

  private void enableInputView(boolean enable) {
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);
    Switch input_switch = findViewById(R.id.switch_connect);
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
