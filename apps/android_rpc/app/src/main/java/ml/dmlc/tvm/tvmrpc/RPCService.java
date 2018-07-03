package ml.dmlc.tvm.tvmrpc;

import android.app.Service;
import android.os.IBinder;
import android.content.Intent;
import android.os.ResultReceiver;
import android.os.Bundle;

public class RPCService extends Service {
    private String m_host;
    private int m_port;
    private String m_key;
    private int m_intent_num;
    private RPCProcessor tvmServerWorker;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        synchronized(this) {
            System.err.println("start command intent");
            m_host = intent.getStringExtra("host");
            m_port = intent.getIntExtra("port", 9090);
            m_key = intent.getStringExtra("key");
            ResultReceiver receiver = intent.getParcelableExtra("receiver");
            System.err.println("got the following: " + m_host + ", " + m_port + ", " + m_key);

            if (tvmServerWorker == null) {
                System.err.println("service created worker...");
                tvmServerWorker = new RPCProcessor();
                tvmServerWorker.setDaemon(true);
                tvmServerWorker.start();
                tvmServerWorker.connect(m_host, m_port, m_key);
            }
            System.err.println("intent num: " + m_intent_num);
            m_intent_num++; 
            Bundle bundle = new Bundle();
            receiver.send(0, bundle);
        }
        // do not restart unless watchdog/app expliciltly does so
        return START_NOT_STICKY;
    }
    
    @Override
    public IBinder onBind(Intent intent) {
        System.err.println("rpc service got onBind, doing nothing...");
        return null;
    }

    @Override
    public void onCreate() {
        System.err.println("rpc service onCreate...");
    }

    @Override
    public void onDestroy() {
        tvmServerWorker.disconnect();
        System.err.println("rpc service onDestroy...");
    }
}
