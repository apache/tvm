package ml.dmlc.tvm.tvmrpc;

import android.app.Service;
import android.os.IBinder;
import android.content.Intent;

public class RPCService extends Service {
    private String host;
    private int port;
    private String key;
    private int intentNum;
    private RPCProcessor tvmServerWorker;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        synchronized(this) {
            System.err.println("start command intent");
            // use an alternate kill to prevent android from recycling the
            // process
            if (intent.getBooleanExtra("kill", false)) {
                System.err.println("rpc service received kill...");
                System.exit(0);
            }
        
            this.host = intent.getStringExtra("host");
            this.port = intent.getIntExtra("port", 9090);
            this.key = intent.getStringExtra("key");
            System.err.println("got the following: " + this.host + ", " + this.port + ", " + this.key);
            System.err.println("intent num: " + this.intentNum);

            if (tvmServerWorker == null) {
                System.err.println("service created worker...");
                tvmServerWorker = new RPCProcessor();
                tvmServerWorker.setDaemon(true);
                tvmServerWorker.start();
                tvmServerWorker.connect(this.host, this.port, this.key);
            }
            else if (tvmServerWorker.timedOut(System.currentTimeMillis())) {
                System.err.println("rpc service timed out, killing self...");
                System.exit(0);
            }
            this.intentNum++; 
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
