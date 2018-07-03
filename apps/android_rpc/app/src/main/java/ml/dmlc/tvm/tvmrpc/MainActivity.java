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
import android.os.ResultReceiver;

import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;
import android.content.Intent;
import android.app.PendingIntent;
import android.app.AlarmManager;


public class MainActivity extends AppCompatActivity {

  private RPCWatchdog watchdog;

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

  class PollResultReceiver extends ResultReceiver {
    private MainActivity activity;

    public PollResultReceiver(Handler handler, MainActivity activity) {
        super(handler);
        this.activity = activity;
    }
    
    @Override
    public void onReceiveResult(int resultCode, Bundle resultData) {
        if (resultCode != 0) {
            System.err.println("abort triggered...");
            System.err.println("creating intent...");
            Intent intent= new Intent(activity, MainActivity.class);
            int hotStart = 1;
            intent.putExtra("hotStart", hotStart);
            System.err.println("creating watchdog shutdown intent...");
            Intent watchdogIntent = new Intent(activity, RPCService.class);
            System.err.println("watchdog shutdown...");
            activity.stopService(watchdogIntent);

            System.err.println("creating pending intent...");
            PendingIntent pendingIntent = PendingIntent.getActivity(activity, 0, intent, PendingIntent.FLAG_CANCEL_CURRENT);
            AlarmManager alarmManager = (AlarmManager) activity.getSystemService(activity.ALARM_SERVICE); 
            alarmManager.set(AlarmManager.RTC, System.currentTimeMillis() + 200, pendingIntent);

            System.err.println("restarting...");
            activity.finishAffinity();
            System.err.println("system exit...");
            System.exit(0); 
        }
    }
  } 

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
          enableInputView(false);
          connectProxy();
        } else {
          disconnect();
          enableInputView(true);
        }
      }
    });
    
    enableInputView(true);

    Intent intent = this.getIntent();
    // detect if app was just restarted
    // if so, connect
    if (intent.getIntExtra("hotStart", 0) == 1) {
        System.err.println("hot start, trying to reconnect...");
        switchConnect.setChecked(true);
    }

  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (watchdog != null) {
        watchdog.disconnect();
        watchdog = null;
    }
  }

  private void connectProxy() {
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);
    final String proxyHost = edProxyAddress.getText().toString();
    final int proxyPort = Integer.parseInt(edProxyPort.getText().toString());
    final String key = edAppKey.getText().toString();

    System.err.println("creating watchdog thread...");
    watchdog = new RPCWatchdog(proxyHost, proxyPort, key, this, new PollResultReceiver(new Handler(), this));
    
    System.err.println("starting watchdog thread...");
    watchdog.start();

    SharedPreferences pref = getApplicationContext().getSharedPreferences("RPCProxyPreference", Context.MODE_PRIVATE);
    SharedPreferences.Editor editor = pref.edit();
    editor.putString("input_address", proxyHost);
    editor.putString("input_port", edProxyPort.getText().toString());
    editor.putString("input_key", key);
    editor.commit();
  }

  private void disconnect() {
    watchdog.disconnect();
    watchdog = null;
  }

  private void enableInputView(boolean enable) {
    EditText edProxyAddress = findViewById(R.id.input_address);
    EditText edProxyPort = findViewById(R.id.input_port);
    EditText edAppKey = findViewById(R.id.input_key);
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
    }
  }
}
