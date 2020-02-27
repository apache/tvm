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

package org.apache.tvm.android.androidcamerademo;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.google.common.util.concurrent.ListenableFuture;

import org.apache.tvm.Function;
import org.apache.tvm.Module;
import org.apache.tvm.NDArray;
import org.apache.tvm.TVMContext;
import org.apache.tvm.TVMType;
import org.apache.tvm.TVMValue;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Vector;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Camera2BasicFragment extends Fragment implements
        ActivityCompat.OnRequestPermissionsResultCallback {
    private static final String TAG = Camera2BasicFragment.class.getSimpleName();
    private static final int PERMISSIONS_REQUEST_CODE = 1;
    // TVM constants
    private static final int OUTPUT_INDEX = 0;
    private static final int IMG_CHANNEL = 3;

    private static final String INPUT_NAME = "data";
    // Configuration values for extraction model. Note that the graph, lib and params is not
    // included with TVM and must be manually placed in the assets/ directory by the user.
    // Graphs and models downloaded from https://github.com/pjreddie/darknet/blob/ may be
    // converted e.g. via  define_and_compile_model.py.
    private static final boolean EXE_GPU = false;
    private static final int MODEL_INPUT_SIZE = 224;
    private static final String MODEL_CL_LIB_FILE = "file:///android_asset/deploy_lib_opencl.so";
    private static final String MODEL_CPU_LIB_FILE = "file:///android_asset/deploy_lib_cpu.so";
    private static final String MODEL_GRAPH_FILE = "file:///android_asset/deploy_graph.json";
    private static final String MODEL_PARAM_FILE = "file:///android_asset/deploy_param.params";
    private static final String MODEL_LABEL_FILE = "file:///android_asset/imagenet.txt";
    private static String[] MODELS;
    private static String mCurModel = "";
    private boolean mRunClassifier = false;
    private int[] mRGBValues = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
    private float[] mCHW = new float[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * IMG_CHANNEL];
    private Semaphore isProcessingDone = new Semaphore(1);
    private boolean mCheckedPermissions = false;
    private AppCompatTextView mResultView;
    private AppCompatTextView mInfoView;
    private ListView mModelView;
    private AssetManager assetManager;
    private Module graphRuntimeModule;
    private Vector<String> labels = new Vector<String>();
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private Camera camera;
    private ImageAnalysis imageAnalysis;
    private ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
            3,
            3,
            1,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>()
    );

    public static Camera2BasicFragment newInstance() {
        return new Camera2BasicFragment();
    }

    private String[] getModelAssets() {
        String[] assetList;
        ArrayList<String> modelAssets = new ArrayList<>();
        String[] modelAssetsArray;
        try {
            assetList = getActivity().getAssets().list("");
        } catch (IOException e) {
            return null;
        }
        for (String asset : assetList) {
            System.err.println("asset: " + asset);
            // hack
            // hack
            if (asset.contains("deploy_lib")) {
                modelAssets.add(asset);
            }
        }
        modelAssetsArray = new String[modelAssets.size()];
        modelAssetsArray = modelAssets.toArray(modelAssetsArray);
        String tmp = modelAssetsArray[modelAssets.size() - 1];
        modelAssetsArray[modelAssets.size() - 1] = modelAssetsArray[0];
        modelAssetsArray[0] = tmp;
        return modelAssetsArray;
    }

    private String[] inference(float[] chw) {
        NDArray inputNdArray = NDArray.empty(new long[]{1, IMG_CHANNEL, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE}, new TVMType("float32"));
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
            PriorityQueue<Integer> pq = new PriorityQueue<>(1000, (Integer idx1, Integer idx2) -> output[idx1] > output[idx2] ? -1 : 1);

            // display the result from extracted output data
            for (int j = 0; j < output.length; ++j) {
                pq.add(j);
            }
            for (int l = 0; l < 5; l++) {
                int idx = pq.poll();
                if (idx < labels.size()) {
                    results[l] = String.format("%.2f", output[idx]) + " : " + labels.get(idx);
                } else {
                    results[l] = "???: unknown";
                }
            }
            return results;
        }
        return new String[5];
    }

    private void updateActiveModel() {
        System.err.println("updating active model...");
        new LoadModelAsyncTask().execute();
    }

    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        if (!mCheckedPermissions && !allPermissionsGranted()) {
            requestPermissions(getRequiredPermissions(), PERMISSIONS_REQUEST_CODE);
            return;
        } else {
            mCheckedPermissions = true;
        }
        mResultView = view.findViewById(R.id.resultTextView);
        mInfoView = view.findViewById(R.id.infoTextView);
        mModelView = view.findViewById(R.id.modelListView);
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
        mModelView.setItemChecked(1, true);
        mModelView.setOnItemClickListener(
                (parent, view1, position, id) -> updateActiveModel());

        new LoadModelAsyncTask().execute();
        System.err.println("view created...");
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        System.err.println("activity created...");
    }

    @Override
    public void onResume() {
        super.onResume();
        System.err.println("on resume...");
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
     * @param assets   The asset manager to be used to load assets.
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
            PackageInfo info = activity
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

    private Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private float[] getFrame(ImageProxy imageProxy) {
        int pixel = 0;
        @SuppressLint("UnsafeExperimentalUsageError") Image image = imageProxy.getImage();
        if(image!=null) {
            Bitmap bitmap = toBitmap(image);
            bitmap.getPixels(mRGBValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        }
        for (int h = 0; h < MODEL_INPUT_SIZE; h++) {
            for (int w = 0; w < MODEL_INPUT_SIZE; w++) {
                int val = mRGBValues[pixel++];
                float r = ((val >> 16) & 0xff) / 255.f;
                float g = ((val >> 8) & 0xff) / 255.f;
                float b = (val & 0xff) / 255.f;
                mCHW[0 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + h * MODEL_INPUT_SIZE + w] = r;
                mCHW[1 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + h * MODEL_INPUT_SIZE + w] = g;
                mCHW[2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + h * MODEL_INPUT_SIZE + w] = b;
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
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        cameraProviderFuture = ProcessCameraProvider.getInstance(getActivity());
    }

    @SuppressLint({"RestrictedApi", "UnsafeExperimentalUsageError"})
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View v = inflater.inflate(R.layout.fragment_camera2_basic, container, false);
        previewView = v.findViewById(R.id.textureView);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future. This should never be reached
            }
        }, ContextCompat.getMainExecutor(getActivity()));

        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(224, 224))
                .setMaxResolution(new Size(800, 800))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(threadPoolExecutor, image -> {
            Log.e(TAG, "w: " + image.getWidth() + " h: " + image.getHeight());

            if (mRunClassifier && isProcessingDone.tryAcquire()) {
                long t1 = SystemClock.uptimeMillis();
                float[] chw = getFrame(image);
                image.close();
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
                isProcessingDone.release();
            }
        });
        return v;
    }

    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        @SuppressLint("RestrictedApi") Preview preview = new Preview.Builder()
                .setMaxResolution(new Size(800, 800))
                .setTargetName("Preview")
                .build();

        preview.setSurfaceProvider(previewView.getPreviewSurfaceProvider());
        CameraSelector cameraSelector =
                new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();
        camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    @Override
    public void onDestroyView() {
        threadPoolExecutor.shutdownNow();
        super.onDestroyView();
    }

    /*
       Load precompiled model on TVM graph runtime and init the system.
    */
    private class LoadModelAsyncTask extends AsyncTask<Void, Void, Integer> {

        @Override
        protected Integer doInBackground(Void... args) {
            mRunClassifier = false;
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
            System.err.println("Uploading compiled function to cache folder");
            try {
                libCacheFilePath = getTempLibFilePath(libFilename);
                byte[] modelLibByte = getBytesFromFile(assetManager, libFilename);
                FileOutputStream fos = new FileOutputStream(libCacheFilePath);
                fos.write(modelLibByte);
                fos.close();
            } catch (IOException e) {
                System.err.println("Problem uploading compiled function!" + e);
                return -1;//failure
            }

            // load parameters
            byte[] modelParams;
            String paramFilename = MODEL_PARAM_FILE.split("file:///android_asset/")[1];
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
            mRunClassifier = true;
            return 0;//success
        }
    }
}
