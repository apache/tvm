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
    }

    @Override
    public void onDestroy() {
        System.err.println("watchdog service onDestroy...");
    }
}
