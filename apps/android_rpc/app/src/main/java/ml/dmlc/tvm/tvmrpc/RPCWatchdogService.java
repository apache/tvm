package ml.dmlc.tvm.tvmrpc;

import android.app.Service;
import android.os.IBinder;
import android.content.Intent;
import android.app.PendingIntent;
import android.app.ActivityManager;
import android.app.Notification;
import android.support.v4.app.NotificationCompat;

public class RPCWatchdogService extends Service {
    private RPCWatchdog watchdog;
    private String host;
    private int port;
    private String key;
    //private int intentNum;
    //private RPCProcessor tvmServerWorker;
    //static final int NOTIFICATION_ID = 420;
    //public static final String ACTION = "ml.dmlc.tvm.tvmrpc.MainActivity";
    //public static final String ANDROID_CHANNEL_ID = "ml.dmlc.tvm.tvmrpc.ANDROID";

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
   //     synchronized(this) {
   //         System.err.println("watchdog start intent");

   //     
   //         //this.host = intent.getStringExtra("host");
   //         //this.port = intent.getIntExtra("port", 9090);
   //         //this.key = intent.getStringExtra("key");
   //         //System.err.println("got the following: " + this.host + ", " + this.port + ", " + this.key);

   //         if (watchdog == null) {
   //             System.err.println("creating watchdog thread...");
   //             watchdog = new RPCWatchdog(this);
   //             System.err.println("starting watchdog thread...");
   //             watchdog.start();
   //         }

   //         //System.err.println("intent num: " + this.intentNum);

   //         //if (tvmServerWorker == null) {
   //         //    System.err.println("service created worker...");
   //         //    tvmServerWorker = new RPCProcessor();
   //         //    tvmServerWorker.setDaemon(true);
   //         //    tvmServerWorker.start();
   //         //    tvmServerWorker.connect(this.host, this.port, this.key);
   //         //}
   //         //else if (tvmServerWorker.timedOut(System.currentTimeMillis())) {
   //         //    System.err.println("rpc service timed out, killing self...");
   //         //    System.exit(0);
   //         //}
   //         //this.intentNum++; 
   //     }
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        System.err.println("watchdog service got onBind, doing nothing...");
        return null;
    }

    @Override
    public void onCreate() {
        System.err.println("watchdog service onCreate...");
        //startServiceWithNotification();
    }

    @Override
    public void onDestroy() {
        System.err.println("watchdog service onDestroy...");
    }

    //private void startServiceWithNotification() {
    //    Intent intent = new Intent(this, MainActivity.class);
    //    intent.setAction(ACTION);
    //    intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);

    //    //PendingIntent contentPendingIntent = PendingIntent.getActivity(this, 0, intent, 0);
    //    System.err.println(getResources().getString(R.string.app_name));
    //    Notification notification = new Notification.Builder(this, ANDROID_CHANNEL_ID)
    //        .setContentTitle(getResources().getString(R.string.app_name))
    //        //.setTicker(getResources().getString(R.string.app_name))
    //        .setContentText("RPC Watchdog Running...")
    //        //.setContentIntent(contentPendingIntent)
    //        //.setOngoing(true)
    //        .build();
    //    //notification.flags = notification.flags | Notification.FLAG_NO_CLEAR;
    //    System.err.println("starting foreground...");
    //    startForeground(1, notification);
    //    System.err.println("started foreground");
    //}
}
