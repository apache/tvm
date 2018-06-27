package ml.dmlc.tvm.tvmrpc;

import android.app.Service;
import android.os.IBinder;
import android.content.Intent;

public class RPCService extends Service {
    private String m_host;
    private int m_port;
    private String m_key;

    private RPCProcessor tvmServerWorker;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        System.err.println("start command intent");
        m_host = intent.getStringExtra("host");
        m_port = intent.getIntExtra("port", 9090);
        m_key = intent.getStringExtra("key");
        System.err.println("got the following...");
        System.err.println(m_host);
        System.err.println(m_port);
        System.err.println(m_key);

        System.err.println("service created worker...");
        tvmServerWorker = new RPCProcessor();
        tvmServerWorker.setDaemon(true);
        tvmServerWorker.start();
        tvmServerWorker.connect(m_host, m_port, m_key);
        
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
