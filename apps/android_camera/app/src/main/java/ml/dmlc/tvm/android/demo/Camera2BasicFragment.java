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

package ml.dmlc.tvm.android.androidcamerademo;

import android.Manifest;
import android.app.Activity;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.PackageInfo;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.media.ImageReader;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.support.v13.app.FragmentCompat;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.view.View;
import android.view.Surface;
import android.view.TextureView;
//import android.view.SurfaceView;
//import android.view.SurfaceHolder;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;


import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;


import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.lang.Byte;
import java.lang.SecurityException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Vector;
import java.util.PriorityQueue;
import java.util.Comparator;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;
import ml.dmlc.tvm.TVMValue;
import ml.dmlc.tvm.TVMType;

public class Camera2BasicFragment extends Fragment implements
        FragmentCompat.OnRequestPermissionsResultCallback {
    private final Object lock = new Object();
    private boolean mRunClassifier = false;
     

    private static final String TAG = MainActivity.class.getSimpleName();

    private static final int PERMISSIONS_REQUEST    = 100;
    private static final String HANDLE_THREAD_NAME = "CameraBackground";

    private static final int PERMISSIONS_REQUEST_CODE = 1;

    private static final int PICTURE_FROM_GALLERY   = 101;
    private static final int PICTURE_FROM_CAMERA    = 102;

    private static final int MAX_PREVIEW_WIDTH = 1920;
    private static final int MAX_PREVIEW_HEIGHT = 1080;

    //private static final int IMAGE_PREVIEW_WIDTH    = 1920;
    //private static final int IMAGE_PREVIEW_HEIGHT   = 1080;

    // TVM constants
    private static final int OUTPUT_INDEX           = 0;
    private static final int IMG_CHANNEL            = 3;
    private static final String INPUT_NAME          = "data";

    // Configuration values for extraction model. Note that the graph, lib and params is not
    // included with TVM and must be manually placed in the assets/ directory by the user.
    // Graphs and models downloaded from https://github.com/pjreddie/darknet/blob/ may be
    // converted e.g. via  define_and_compile_model.py.
    private static final boolean EXE_GPU            = false;
    private static final int MODEL_INPUT_SIZE       = 224;
    private static final String MODEL_CL_LIB_FILE   = "file:///android_asset/deploy_lib_opencl.so";
    private static final String MODEL_CPU_LIB_FILE  = "file:///android_asset/deploy_lib_cpu.so";
    private static final String MODEL_GRAPH_FILE    = "file:///android_asset/deploy_graph.json";
    private static final String MODEL_PARAM_FILE    = "file:///android_asset/deploy_param.params";
    private static final String MODEL_LABEL_FILE    = "file:///android_asset/imagenet.txt";
    private static String[] MODELS;
    private static String mCurModel = "";

    private int[] mRGBValues = new int[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE];
    private float[] mCHW = new float[MODEL_INPUT_SIZE*MODEL_INPUT_SIZE*IMG_CHANNEL];

    //private PreviewCallback mPreviewCallback;
    private String mCameraId;
    private boolean mCheckedPermissions = false;
    private RenderScript mRenderScript;
    private ScriptIntrinsicYuvToRGB mScriptIntrinsicYuvToRGB;
    private Type.Builder mYuvType;
    private Type.Builder mRGBType;

    private CaptureRequest.Builder mPreviewRequestBuilder;
    private CaptureRequest mPreviewRequest;

    private Semaphore mCameraOpenCloseLock = new Semaphore(1);
    private CameraDevice mCameraDevice;
    private CameraCaptureSession mCaptureSession;
    private Size mPreviewSize;
    private CameraCaptureSession.CaptureCallback mCaptureCallback =
      new CameraCaptureSession.CaptureCallback() {

        @Override
        public void onCaptureProgressed(
            @NonNull CameraCaptureSession session,
            @NonNull CaptureRequest request,
            @NonNull CaptureResult partialResult) {}

        @Override
        public void onCaptureCompleted(
            @NonNull CameraCaptureSession session,
            @NonNull CaptureRequest request,
            @NonNull TotalCaptureResult result) {}
      };

    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
    private ImageReader mImageReader;

    private AutoFitTextureView mAutoFitTextureView;

    private TextView mResultView;
    private TextView mInfoView;
    private ListView mModelView;
    private AssetManager assetManager;
    private Module graphRuntimeModule;
    private Vector<String> labels = new Vector<String>();

    private String[] getModelAssets() {
        String[] assetList;
        ArrayList<String> modelAssets = new ArrayList<String>();
        String[] modelAssetsArray;
        try {
        assetList = getActivity().getAssets().list("");
        } catch (IOException e) {
            return null;
        }
        for (String asset: assetList) {
            System.err.println("asset: " + asset);
            // hack
            if (asset.indexOf("webkit") < 0 && asset.indexOf("image") < 0  && asset.indexOf("py") < 0)           {
                modelAssets.add(asset); 
            }
        }
        modelAssetsArray = new String[modelAssets.size()];
        modelAssetsArray = modelAssets.toArray(modelAssetsArray);
        String tmp = modelAssetsArray[modelAssets.size()-1];
        modelAssetsArray[modelAssets.size()-1] = modelAssetsArray[0];
        modelAssetsArray[0] = tmp;
        return modelAssetsArray;
    }

    private Runnable mPeriodicClassify = 
        new Runnable() {
            @Override
            public void run() {
                synchronized (lock) {
                    if (mRunClassifier) {
                        long t1 = SystemClock.uptimeMillis();
                        float [] chw = getFrame();
                        long t2 = SystemClock.uptimeMillis(); 
                        String[] results = inference(chw);
                        long t3 = SystemClock.uptimeMillis();
                        String msg = "";
                        for (int l = 1; l < 5; l++) {
                            msg = msg + results[l] + "\n";
                        }
                        msg += "getFrame(): " + (t2 - t1) + "ms" + "\n";
                        msg += "inference(): " + (t3 - t2) + "ms" + "\n";

                        mResultView.setText("model: " + mCurModel + "\n" + results[0]);
                        mInfoView.setText(msg);
                    }
                }
                mBackgroundHandler.post(mPeriodicClassify);
            }
        };
    
    // tflite style
    private final TextureView.SurfaceTextureListener mSurfaceTextureListener = 
        new TextureView.SurfaceTextureListener() {
        
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
            System.err.println("surface texture available...");
            openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
            System.err.println("surface texture size changed...");
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture texture) {}
    };

  private void StartBackgroundThread() {
    System.err.println("starting background thread...");
    //mBackgroundHandler = new Handler
  }

  private void createCameraPreviewSession() {
    try {
      SurfaceTexture texture = mAutoFitTextureView.getSurfaceTexture();
      assert texture != null;

      // We configure the size of default buffer to be the size of camera preview we want.
      texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());

      // This is the output Surface we need to start preview.
      Surface surface = new Surface(texture);

      // We set up a CaptureRequest.Builder with the output Surface.
      mPreviewRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
      mPreviewRequestBuilder.addTarget(surface);

      // Here, we create a CameraCaptureSession for camera preview.
      mCameraDevice.createCaptureSession(
          Arrays.asList(surface),
          new CameraCaptureSession.StateCallback() {

            @Override
            public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
              // The camera is already closed
              if (null == mCameraDevice) {
                return;
              }

              // When the session is ready, we start displaying the preview.
              mCaptureSession = cameraCaptureSession;
              try {
                // Auto focus should be continuous for camera preview.
                mPreviewRequestBuilder.set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                // Finally, we start displaying the camera preview.
                mPreviewRequest = mPreviewRequestBuilder.build();
                mCaptureSession.setRepeatingRequest(
                    mPreviewRequest, mCaptureCallback, mBackgroundHandler);
              } catch (CameraAccessException e) {
                Log.e(TAG, "Failed to set up config to capture Camera", e);
              }
            }

            @Override
            public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
              Log.e(TAG, "onConfigureFailed.");
            }
          },
          null);
    } catch (CameraAccessException e) {
      Log.e(TAG, "Failed to preview Camera", e);
    }
  }

private void setUpCameraOutputs(int width, int height) {
    Activity activity = getActivity();
    CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
    try {
      for (String cameraId : manager.getCameraIdList()) {
        System.err.println("id: " + cameraId);
        CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          continue;
        }

        StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) {
          continue;
        }
        System.err.println("map");
        // // For still image captures, we use the largest available size.
        Size largest =
            Collections.max(
                Arrays.asList(map.getOutputSizes(ImageFormat.JPEG)), new CompareSizesByArea());
        mImageReader =
            ImageReader.newInstance(
                largest.getWidth(), largest.getHeight(), ImageFormat.JPEG, /*maxImages*/ 2);

        // Find out if we need to swap dimension to get the preview size relative to sensor
        // coordinate.
        int displayRotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        // noinspection ConstantConditions
        /* Orientation of the camera sensor */
        int sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        boolean swappedDimensions = false;
        switch (displayRotation) {
          case Surface.ROTATION_0:
          case Surface.ROTATION_180:
            if (sensorOrientation == 90 || sensorOrientation == 270) {
              swappedDimensions = true;
            }
            break;
          case Surface.ROTATION_90:
          case Surface.ROTATION_270:
            if (sensorOrientation == 0 || sensorOrientation == 180) {
              swappedDimensions = true;
            }
            break;
          default:
            Log.e(TAG, "Display rotation is invalid: " + displayRotation);
        }

        Point displaySize = new Point();
        System.err.println("point");
        activity.getWindowManager().getDefaultDisplay().getSize(displaySize);
        int rotatedPreviewWidth = width;
        int rotatedPreviewHeight = height;
        int maxPreviewWidth = displaySize.x;
        int maxPreviewHeight = displaySize.y;

        if (swappedDimensions) {
          rotatedPreviewWidth = height;
          rotatedPreviewHeight = width;
          maxPreviewWidth = displaySize.y;
          maxPreviewHeight = displaySize.x;
        }

        if (maxPreviewWidth > MAX_PREVIEW_WIDTH) {
          maxPreviewWidth = MAX_PREVIEW_WIDTH;
        }

        if (maxPreviewHeight > MAX_PREVIEW_HEIGHT) {
          maxPreviewHeight = MAX_PREVIEW_HEIGHT;
        }

        mPreviewSize =
            chooseOptimalSize(
                map.getOutputSizes(SurfaceTexture.class),
                rotatedPreviewWidth,
                rotatedPreviewHeight,
                maxPreviewWidth,
                maxPreviewHeight,
                largest);
        System.err.println("optimal size...");
        // We fit the aspect ratio of TextureView to the size of preview we picked.
        int orientation = getResources().getConfiguration().orientation;
        if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
          mAutoFitTextureView.setAspectRatio(mPreviewSize.getWidth(), mPreviewSize.getHeight());
        } else {
          mAutoFitTextureView.setAspectRatio(mPreviewSize.getHeight(), mPreviewSize.getWidth());
        }

        this.mCameraId = cameraId;
        return;
      }
    } catch (CameraAccessException e) {
      Log.e(TAG, "Failed to access Camera", e);
    } catch (NullPointerException e) {
      // Currently an NPE is thrown when the Camera2API is used but not supported on the
      // device this code runs.
      e.printStackTrace();
      System.err.println("NPE Camera2API");
    }
  }


  private void openCamera(int width, int height) {
    if (!mCheckedPermissions && !allPermissionsGranted()) {
      FragmentCompat.requestPermissions(this, getRequiredPermissions(), PERMISSIONS_REQUEST_CODE);
      return;
    } else {
      mCheckedPermissions = true;
    }
    setUpCameraOutputs(width, height);
    configureTransform(width, height);
    Activity activity = getActivity();
    CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
    try {
      if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
        throw new RuntimeException("Time out waiting to lock camera opening.");
      }
      manager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
    } catch (CameraAccessException e) {
      Log.e(TAG, "Failed to open Camera", e);
    } catch (SecurityException e) {
      Log.e(TAG, "Failed to open Camera (security)", e);
    } catch (InterruptedException e) {
      throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
    }
  }

   private final CameraDevice.StateCallback mStateCallback = 
        new CameraDevice.StateCallback() {
        
        @Override
        public void onOpened(@NonNull CameraDevice currentCameraDevice) {
            mCameraOpenCloseLock.release();
            mCameraDevice = currentCameraDevice;
            createCameraPreviewSession();    
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice currentCameraDevice) {
            mCameraOpenCloseLock.release();
            currentCameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice currentCameraDevice, int error) {
            mCameraOpenCloseLock.release();
            currentCameraDevice.close();
            mCameraDevice = null;
            Activity activity = getActivity();
            if (activity != null)
                activity.finish();
        }

      };

  private static class CompareSizesByArea implements Comparator<Size> {

    @Override
    public int compare(Size lhs, Size rhs) {
      // We cast here to ensure the multiplications won't overflow
      return Long.signum(
          (long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
    }
  }

  private static Size chooseOptimalSize(
      Size[] choices,
      int textureViewWidth,
      int textureViewHeight,
      int maxWidth,
      int maxHeight,
      Size aspectRatio) {

    // Collect the supported resolutions that are at least as big as the preview Surface
    List<Size> bigEnough = new ArrayList<>();
    // Collect the supported resolutions that are smaller than the preview Surface
    List<Size> notBigEnough = new ArrayList<>();
    int w = aspectRatio.getWidth();
    int h = aspectRatio.getHeight();
    for (Size option : choices) {
      if (option.getWidth() <= maxWidth
          && option.getHeight() <= maxHeight
          && option.getHeight() == option.getWidth() * h / w) {
        if (option.getWidth() >= textureViewWidth && option.getHeight() >= textureViewHeight) {
          bigEnough.add(option);
        } else {
          notBigEnough.add(option);
        }
      }
    }

    // Pick the smallest of those big enough. If there is no one big enough, pick the
    // largest of those not big enough.
    if (bigEnough.size() > 0) {
      return Collections.min(bigEnough, new CompareSizesByArea());
    } else if (notBigEnough.size() > 0) {
      return Collections.max(notBigEnough, new CompareSizesByArea());
    } else {
      Log.e(TAG, "Couldn't find any suitable preview size");
      return choices[0];
    }
  }

    private String[] inference(float[] chw) {
        NDArray inputNdArray = NDArray.empty(new long[]{1, IMG_CHANNEL, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE}, new TVMType("float32"));;
        inputNdArray.copyFrom(chw);
        Function setInputFunc = graphRuntimeModule.getFunction("set_input");
        setInputFunc.pushArg(INPUT_NAME).pushArg(inputNdArray).invoke();
        // release tvm local variables
        inputNdArray.release();
        setInputFunc.release();

        // get the function from the module(run it)
        Function runFunc = graphRuntimeModule.getFunction("run");
        runFunc.invoke();
        // release tvm local variables
        runFunc.release();

        // get the function from the module(get output data)
        NDArray outputNdArray = NDArray.empty(new long[]{1, 1000}, new TVMType("float32"));
        Function getOutputFunc = graphRuntimeModule.getFunction("get_output");
        getOutputFunc.pushArg(OUTPUT_INDEX).pushArg(outputNdArray).invoke();
        float[] output = outputNdArray.asFloatArray();
        // release tvm local variables
        outputNdArray.release();
        getOutputFunc.release();

        if (null != output) {
            String[] results = new String[5];
            // top-5
            PriorityQueue<Integer> pq = new PriorityQueue<Integer>(1000, (Integer idx1, Integer idx2) -> output[idx1] > output[idx2] ? -1 : 1);
        
            // display the result from extracted output data
            for (int j = 0; j < output.length; ++j) {
                pq.add(j);
              }
            for (int l = 0; l < 5; l++) {
                int idx = pq.poll();
                if (idx < labels.size()) {
                    results[l] = String.format("%.2f", output[idx]) + " : " +  labels.get(idx);
                } else {
                    results[l] = "???: unknown";
                }
            }
            return results;
        }
        return new String[5];
    }
    
    public static Camera2BasicFragment newInstance() {
        return new Camera2BasicFragment();
    }

    private void updateActiveModel() {
        System.err.println("updating active model...");
        new LoadModelAsyncTask().execute();
    }

    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        if (!mCheckedPermissions && !allPermissionsGranted()) {
         FragmentCompat.requestPermissions(this, getRequiredPermissions(), PERMISSIONS_REQUEST_CODE);
              return;
        } else {
          mCheckedPermissions = true;
        } 
        mAutoFitTextureView = (AutoFitTextureView) view.findViewById(R.id.textureView);
        mResultView = (TextView) view.findViewById(R.id.resultTextView);
        mInfoView = (TextView) view.findViewById(R.id.infoTextView);
        mModelView = (ListView) view.findViewById(R.id.modelListView);
        if (assetManager == null) {
          assetManager = getActivity().getAssets();
        }

        mModelView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
        MODELS = getModelAssets();
        System.err.println(MODELS);
    
        ArrayAdapter<String> modelAdapter =
        new ArrayAdapter<>(
                    getContext(), R.layout.listview_row, R.id.listview_row_text, MODELS);
            mModelView.setAdapter(modelAdapter);
            mModelView.setItemChecked(0, true);
            mModelView.setOnItemClickListener(
                new AdapterView.OnItemClickListener() {
                  @Override
                  public void onItemClick(AdapterView<?> parent, View view, int
        position, long id) {
                    updateActiveModel();
                  }
                });

        new LoadModelAsyncTask().execute();
        System.err.println("view created...");
    }

  /** Starts a background thread and its {@link Handler}. */
  private void startBackgroundThread() {
    mBackgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    // Start the classification train & load an initial model.
    synchronized (lock) {
      mRunClassifier = true;
    }
    mBackgroundHandler.post(mPeriodicClassify);
  }


  @Override
  public void onActivityCreated(Bundle savedInstanceState) {
    super.onActivityCreated(savedInstanceState);
    System.err.println("activity created..."); 
  }

  /*
     Load precompiled model on TVM graph runtime and init the system.
  */
  private class LoadModelAsyncTask extends AsyncTask<Void, Void, Integer> {

    @Override
    protected Integer doInBackground(Void... args) {
      // load synset name
      int modelIndex = mModelView.getCheckedItemPosition();
      String model = MODELS[modelIndex];
      
      String labelFilename = MODEL_LABEL_FILE.split("file:///android_asset/")[1];
      System.err.println("Reading synset name from: " + labelFilename);
      try {
        String labelsContent = new String(getBytesFromFile(assetManager, labelFilename));
        for (String line : labelsContent.split("\\r?\\n")) {
            labels.add(line);
        }
      } catch (IOException e) {
        System.err.println("Problem reading synset name file!" + e);
        return -1;//failure
      }

      // load json graph
      String modelGraph = null;
      String graphFilename = MODEL_GRAPH_FILE.split("file:///android_asset/")[1];
      graphFilename = model + "/" + graphFilename;
      System.err.println("Reading json graph from: " + graphFilename);
      try {
        modelGraph = new String(getBytesFromFile(assetManager, graphFilename));
      } catch (IOException e) {
        System.err.println("Problem reading json graph file!" + e);
        return -1;//failure
      }

      // upload tvm compiled function on application cache folder
      String libCacheFilePath = null;
      String libFilename = EXE_GPU ? MODEL_CL_LIB_FILE.split("file:///android_asset/")[1] :
              MODEL_CPU_LIB_FILE.split("file:///android_asset/")[1];
      String libPath = model + "/" + libFilename;
      System.err.println("Uploading compiled function to cache folder");
      try {
        libCacheFilePath = getTempLibFilePath(libFilename);
        byte[] modelLibByte = getBytesFromFile(assetManager, libPath);
        FileOutputStream fos = new FileOutputStream(libCacheFilePath);
        fos.write(modelLibByte);
        fos.close();
      } catch (IOException e) {
        System.err.println("Problem uploading compiled function!" + e);
        return -1;//failure
      }

      // load parameters
      byte[] modelParams = null;
      String paramFilename = MODEL_PARAM_FILE.split("file:///android_asset/")[1];
      paramFilename = model + "/" + paramFilename;
      try {
        modelParams = getBytesFromFile(assetManager, paramFilename);
      } catch (IOException e) {
        System.err.println("Problem reading params file!" + e);
        return -1;//failure
      }

      System.err.println("creating java tvm context...");
      // create java tvm context
      TVMContext tvmCtx = EXE_GPU ? TVMContext.opencl() : TVMContext.cpu();
      
      System.err.println("loading compiled functions...");
      System.err.println(libCacheFilePath);
      // tvm module for compiled functions
      Module modelLib = Module.load(libCacheFilePath);
      
      
      // get global function module for graph runtime
      System.err.println("getting graph runtime create handle...");

      Function runtimeCreFun = Function.getFunction("tvm.graph_runtime.create");
      System.err.println("creating graph runtime...");
      
      System.err.println("ctx type: " + tvmCtx.deviceType);
      System.err.println("ctx id: " + tvmCtx.deviceId);

      
      synchronized(lock) { 
        TVMValue runtimeCreFunRes = runtimeCreFun.pushArg(modelGraph)
                .pushArg(modelLib)
                .pushArg(tvmCtx.deviceType)
                .pushArg(tvmCtx.deviceId)
                .invoke();

        System.err.println("as module...");
        graphRuntimeModule = runtimeCreFunRes.asModule();
        System.err.println("getting graph runtime load params handle...");
        // get the function from the module(load parameters)
        Function loadParamFunc = graphRuntimeModule.getFunction("load_params");
        System.err.println("loading params...");
        loadParamFunc.pushArg(modelParams).invoke();
        // release tvm local variables
        modelLib.release();
        loadParamFunc.release();
        runtimeCreFun.release();

        mCurModel = model;
       } 
       return 0;//success
    }

    @Override
    protected void onPreExecute() {
      super.onPreExecute();
    }

    @Override
    protected void onPostExecute(Integer status) {
      if (status < 0) {
        System.err.println("error status" + status);
        System.err.println("Fail to initialized model, check compiled model");
      } else if (status > 0) {
        System.err.println("debug status" + status); 
      } else {
        System.err.println("finished pre..."); 
      }
      startBackgroundThread();
    }

  }

  @Override
  public void onResume() {
    super.onResume();
    System.err.println("on resume...");
    if (mAutoFitTextureView.isAvailable()) {
      System.err.println("autofittextureview available...");
      openCamera(mAutoFitTextureView.getWidth(), mAutoFitTextureView.getHeight());
    } else {
      mAutoFitTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
    }
  }

  @Override
  public void onDestroy() {
    // release tvm local variables
    if (null != graphRuntimeModule)
      graphRuntimeModule.release();
    super.onDestroy();
  }

  /**
   * Read file from assets and return byte array.
   *
   * @param assets The asset manager to be used to load assets.
   * @param fileName The filepath of read file.
   * @return byte[] file content
   * @throws IOException
   */
  private byte[] getBytesFromFile(AssetManager assets, String fileName) throws IOException {
    InputStream is = assets.open(fileName);
    int length = is.available();
    byte[] bytes = new byte[length];
    // Read in the bytes
    int offset = 0;
    int numRead = 0;
    try {
      while (offset < bytes.length
            && (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
        offset += numRead;
      }
    } finally {
        is.close();
    }
    // Ensure all the bytes have been read in
    if (offset < bytes.length) {
        throw new IOException("Could not completely read file " + fileName);
    }
    return bytes;
  }

  /**
   * Get application cache path where to place compiled functions.
   *
   * @param fileName library file name.
   * @return String application cache folder path
   * @throws IOException
   */
  private final String getTempLibFilePath(String fileName) throws IOException {
    File tempDir = File.createTempFile("tvm4j_demo_", "");
    if (!tempDir.delete() || !tempDir.mkdir()) {
      throw new IOException("Couldn't create directory " + tempDir.getAbsolutePath());
    }
    return (tempDir + File.separator + fileName);
  }

  private String[] getRequiredPermissions() {
    Activity activity = getActivity();
    try {
      PackageInfo info =
          activity
              .getPackageManager()
              .getPackageInfo(activity.getPackageName(), PackageManager.GET_PERMISSIONS);
      String[] ps = info.requestedPermissions;
      if (ps != null && ps.length > 0) {
        return ps;
      } else {
        return new String[0];
      }
    } catch (Exception e) {
      return new String[0];
    }
  }

  private float[] getFrame() {
    Bitmap bitmap = mAutoFitTextureView.getBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    bitmap.getPixels(mRGBValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    int pixel = 0;
    for (int h = 0; h < MODEL_INPUT_SIZE; h++) {
      for (int w = 0; w < MODEL_INPUT_SIZE; w++) {
        int val = mRGBValues[pixel++];
        float r = ((val >> 16) & 0xff) / 255.f;
        float g = ((val >> 8) & 0xff) / 255.f;
        float b = (val & 0xff) / 255.f;
        mCHW[0*MODEL_INPUT_SIZE*MODEL_INPUT_SIZE + h*MODEL_INPUT_SIZE + w] = r;
        mCHW[1*MODEL_INPUT_SIZE*MODEL_INPUT_SIZE + h*MODEL_INPUT_SIZE + w] = g;
        mCHW[2*MODEL_INPUT_SIZE*MODEL_INPUT_SIZE + h*MODEL_INPUT_SIZE + w] = b;
      }
    } 
    return mCHW;
  }

  private boolean allPermissionsGranted() {
    for (String permission : getRequiredPermissions()) {
      if (ContextCompat.checkSelfPermission(getActivity(), permission)
          != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
  }

  @Override
  public View onCreateView(
      LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
    return inflater.inflate(R.layout.fragment_camera2_basic, container, false);
  }

  private void configureTransform(int viewWidth, int viewHeight) {
    Activity activity = getActivity();
    if (null == mAutoFitTextureView || null == mPreviewSize || null == activity) {
      return;
    }
    int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
    Matrix matrix = new Matrix();
    RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
    RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
    float centerX = viewRect.centerX();
    float centerY = viewRect.centerY();
    if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
      bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
      matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
      float scale =
          Math.max(
              (float) viewHeight / mPreviewSize.getHeight(),
              (float) viewWidth / mPreviewSize.getWidth());
      matrix.postScale(scale, scale, centerX, centerY);
      matrix.postRotate(90 * (rotation - 2), centerX, centerY);
    } else if (Surface.ROTATION_180 == rotation) {
      matrix.postRotate(180, centerX, centerY);
    }
    mAutoFitTextureView.setTransform(matrix);
  }
}
