package ml.dmlc.tvm.tvmrpc;

import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.EditText;

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
  }

  private Server tvmServer;
  public void connectProxy(View view) {
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

  public void disconnect(View view) {
    if (tvmServer != null) {
      tvmServer.terminate();
      System.err.println("Disconnected.");
    }
  }
}
