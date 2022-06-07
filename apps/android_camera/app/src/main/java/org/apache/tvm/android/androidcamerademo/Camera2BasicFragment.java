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
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
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
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.renderscript.Allocation;
import androidx.renderscript.Element;
import androidx.renderscript.RenderScript;
import androidx.renderscript.Script;
import androidx.renderscript.Type;

import com.google.common.util.concurrent.ListenableFuture;

import org.apache.tvm.Function;
import org.apache.tvm.Module;
import org.apache.tvm.NDArray;
import org.apache.tvm.Device;
import org.apache.tvm.TVMType;
import org.apache.tvm.TVMValue;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Camera2BasicFragment extends Fragment {
    private static final String TAG = Camera2BasicFragment.class.getSimpleName();

    // TVM constants
    private static final int OUTPUT_INDEX = 0;
    private static final int IMG_CHANNEL = 3;
    private static final boolean EXE_GPU = false;
    private static final int MODEL_INPUT_SIZE = 224;
    private static final String MODEL_CL_LIB_FILE = "deploy_lib_opencl.so";
    private static final String MODEL_CPU_LIB_FILE = "deploy_lib_cpu.so";
    private static final String MODEL_GRAPH_FILE = "deploy_graph.json";
    private static final String MODEL_PARAM_FILE = "deploy_param.params";
    private static final String MODEL_LABEL_FILE = "image_net_labels.json";
    private static final String MODELS = "models";
    private static String INPUT_NAME = "input_1";
    private static String[] models;
    private static String mCurModel = "";
    private final float[] mCHW = new float[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * IMG_CHANNEL];
    private final float[] mCHW2 = new float[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * IMG_CHANNEL];
    private final Semaphore isProcessingDone = new Semaphore(1);
    private final ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
            3,
            3,
            1,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>()
    );
    // rs creation just for demo. Create rs just once in onCreate and use it again.
    private RenderScript rs;
    private ScriptC_yuv420888 mYuv420;
    private boolean mRunClassifier = false;

    private AppCompatTextView mResultView;
    private AppCompatTextView mInfoView;
    private ListView mModelView;
    private AssetManager assetManager;
    private Module graphExecutorModule;
    private JSONObject labels;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private ImageAnalysis imageAnalysis;

    static Camera2BasicFragment newInstance() {
        return new Camera2BasicFragment();
    }

    private static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                Log.w(TAG, "Rotation of %d % 90 != 0 " + applyRotation);
            }

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }

    private String[] getModels() {
        String[] models;
        try {
            models = getActivity().getAssets().list(MODELS);
        } catch (IOException e) {
            return null;
        }
        return models;
    }

    @SuppressLint("DefaultLocale")
    private String[] inference(float[] chw) {
        NDArray inputNdArray = NDArray.empty(new long[]{1, IMG_CHANNEL, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE}, new TVMType("float32"));
        inputNdArray.copyFrom(chw);
        Function setInputFunc = graphExecutorModule.getFunction("set_input");
        setInputFunc.pushArg(INPUT_NAME).pushArg(inputNdArray).invoke();
        // release tvm local variables
        inputNdArray.release();
        setInputFunc.release();

        // get the function from the module(run it)
        Function runFunc = graphExecutorModule.getFunction("run");
        runFunc.invoke();
        // release tvm local variables
        runFunc.release();

        // get the function from the module(get output data)
        NDArray outputNdArray = NDArray.empty(new long[]{1, 1000}, new TVMType("float32"));
        Function getOutputFunc = graphExecutorModule.getFunction("get_output");
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
                //noinspection ConstantConditions
                int idx = pq.poll();
                if (idx < labels.length()) {
                    try {
                        results[l] = String.format("%.2f", output[idx]) + " : " + labels.getString(Integer.toString(idx));
                    } catch (JSONException e) {
                        Log.e(TAG, "index out of range", e);
                    }
                } else {
                    results[l] = "???: unknown";
                }
            }
            return results;
        }
        return new String[5];
    }

    private void updateActiveModel() {
        Log.i(TAG, "updating active model...");
        new LoadModelAsyncTask().execute();
    }

    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {

        mResultView = view.findViewById(R.id.resultTextView);
        mInfoView = view.findViewById(R.id.infoTextView);
        mModelView = view.findViewById(R.id.modelListView);
        if (assetManager == null) {
            assetManager = getActivity().getAssets();
        }

        mModelView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
        models = getModels();

        ArrayAdapter<String> modelAdapter =
                new ArrayAdapter<>(
                        getContext(), R.layout.listview_row, R.id.listview_row_text, models);
        mModelView.setAdapter(modelAdapter);
        mModelView.setItemChecked(0, true);
        mModelView.setOnItemClickListener(
                (parent, view1, position, id) -> updateActiveModel());

        new LoadModelAsyncTask().execute();
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
    }

    @Override
    public void onDestroy() {
        // release tvm local variables
        if (null != graphExecutorModule)
            graphExecutorModule.release();
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
        int numRead;
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
    private String getTempLibFilePath(String fileName) throws IOException {
        File tempDir = File.createTempFile("tvm4j_demo_", "");
        if (!tempDir.delete() || !tempDir.mkdir()) {
            throw new IOException("Couldn't create directory " + tempDir.getAbsolutePath());
        }
        return (tempDir + File.separator + fileName);
    }

    private Bitmap YUV_420_888_toRGB(Image image, int width, int height) {
        // Get the three image planes
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        byte[] y = new byte[buffer.remaining()];
        buffer.get(y);

        buffer = planes[1].getBuffer();
        byte[] u = new byte[buffer.remaining()];
        buffer.get(u);

        buffer = planes[2].getBuffer();
        byte[] v = new byte[buffer.remaining()];
        buffer.get(v);

        int yRowStride = planes[0].getRowStride();
        int uvRowStride = planes[1].getRowStride();
        int uvPixelStride = planes[1].getPixelStride();


        // Y,U,V are defined as global allocations, the out-Allocation is the Bitmap.
        // Note also that uAlloc and vAlloc are 1-dimensional while yAlloc is 2-dimensional.
        Type.Builder typeUcharY = new Type.Builder(rs, Element.U8(rs));
        typeUcharY.setX(yRowStride).setY(height);
        Allocation yAlloc = Allocation.createTyped(rs, typeUcharY.create());
        yAlloc.copyFrom(y);
        mYuv420.set_ypsIn(yAlloc);

        Type.Builder typeUcharUV = new Type.Builder(rs, Element.U8(rs));
        // note that the size of the u's and v's are as follows:
        //      (  (width/2)*PixelStride + padding  ) * (height/2)
        // =    (RowStride                          ) * (height/2)
        typeUcharUV.setX(u.length);
        Allocation uAlloc = Allocation.createTyped(rs, typeUcharUV.create());
        uAlloc.copyFrom(u);
        mYuv420.set_uIn(uAlloc);

        Allocation vAlloc = Allocation.createTyped(rs, typeUcharUV.create());
        vAlloc.copyFrom(v);
        mYuv420.set_vIn(vAlloc);

        // handover parameters
        mYuv420.set_picWidth(width);
        mYuv420.set_uvRowStride(uvRowStride);
        mYuv420.set_uvPixelStride(uvPixelStride);

        Bitmap outBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Allocation outAlloc = Allocation.createFromBitmap(rs, outBitmap, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_SCRIPT);

        Script.LaunchOptions lo = new Script.LaunchOptions();
        lo.setX(0, width);  // by this we ignore the y’s padding zone, i.e. the right side of x between width and yRowStride
        lo.setY(0, height);

        mYuv420.forEach_doConvert(outAlloc, lo);
        outAlloc.copyTo(outBitmap);

        return outBitmap;
    }

    private float[] getFrame(ImageProxy imageProxy) {
        @SuppressLint("UnsafeOptInUsageError")
        Image image = imageProxy.getImage();
        // extract the jpeg content
        if (image == null) {
            return null;
        }
        Bitmap imageBitmap = YUV_420_888_toRGB(image, image.getWidth(), image.getHeight());

        imageProxy.close();
        // crop input image at centre to model input size
        Bitmap cropImageBitmap = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Matrix frameToCropTransform = getTransformationMatrix(imageBitmap.getWidth(), imageBitmap.getHeight(),
                MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 0, true);
        Canvas canvas = new Canvas(cropImageBitmap);
        canvas.drawBitmap(imageBitmap, frameToCropTransform, null);

        // image pixel int values
        int[] pixelValues = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
        // image RGB float values

        // pre-process the image data from 0-255 int to normalized float based on the
        // provided parameters.
        cropImageBitmap.getPixels(pixelValues, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
        for (int j = 0; j < pixelValues.length; ++j) {
            mCHW2[j * 3 + 0] = ((pixelValues[j] >> 16) & 0xFF) / 255.0f;
            mCHW2[j * 3 + 1] = ((pixelValues[j] >> 8) & 0xFF) / 255.0f;
            mCHW2[j * 3 + 2] = (pixelValues[j] & 0xFF) / 255.0f;
        }

        // pre-process the image rgb data transpose based on the provided parameters.
        for (int k = 0; k < IMG_CHANNEL; ++k) {
            for (int l = 0; l < MODEL_INPUT_SIZE; ++l) {
                for (int m = 0; m < MODEL_INPUT_SIZE; ++m) {
                    int dst_index = m + MODEL_INPUT_SIZE * l + MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * k;
                    int src_index = k + IMG_CHANNEL * m + IMG_CHANNEL * MODEL_INPUT_SIZE * l;
                    mCHW[dst_index] = mCHW2[src_index];
                }
            }
        }

        return mCHW;
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
        rs = RenderScript.create(getActivity());
        mYuv420 = new ScriptC_yuv420888(rs);
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
                .setMaxResolution(new Size(300, 300))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(threadPoolExecutor, image -> {
            Log.e(TAG, "w: " + image.getWidth() + " h: " + image.getHeight());
            if (mRunClassifier && isProcessingDone.tryAcquire()) {
                long t1 = SystemClock.uptimeMillis();
                //float[] chw = getFrame(image);
                //float[] chw = YUV_420_888_toRGBPixels(image);
                float[] chw = getFrame(image);
                if (chw != null) {
                    long t2 = SystemClock.uptimeMillis();
                    String[] results = inference(chw);
                    long t3 = SystemClock.uptimeMillis();
                    StringBuilder msgBuilder = new StringBuilder();
                    for (int l = 1; l < 5; l++) {
                        msgBuilder.append(results[l]).append("\n");
                    }
                    String msg = msgBuilder.toString();
                    msg += "getFrame(): " + (t2 - t1) + "ms" + "\n";
                    msg += "inference(): " + (t3 - t2) + "ms" + "\n";
                    String finalMsg = msg;
                    this.getActivity().runOnUiThread(() -> {
                        mResultView.setText(String.format("model: %s \n %s", mCurModel, results[0]));
                        mInfoView.setText(finalMsg);
                    });
                }
                isProcessingDone.release();
            }
            image.close();
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
        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    @Override
    public void onDestroyView() {
        threadPoolExecutor.shutdownNow();
        super.onDestroyView();
    }

    private void setInputName(String modelName) {
        if (modelName.equals("mobilenet_v2")) {
            INPUT_NAME = "input_1";
        } else if (modelName.equals("resnet18_v1")) {
            INPUT_NAME = "data";
        } else {
            throw new RuntimeException("Model input may not be right. Please set INPUT_NAME here explicitly.");
        }
    }

    /*
       Load precompiled model on TVM graph executor and init the system.
    */
    private class LoadModelAsyncTask extends AsyncTask<Void, Void, Integer> {

        @Override
        protected Integer doInBackground(Void... args) {
            mRunClassifier = false;
            // load synset name
            int modelIndex = mModelView.getCheckedItemPosition();
            setInputName(models[modelIndex]);
            String model = MODELS + "/" + models[modelIndex];
            String labelFilename = MODEL_LABEL_FILE;
            Log.i(TAG, "Reading labels from: " + model + "/" + labelFilename);
            try {
                labels = new JSONObject(new String(getBytesFromFile(assetManager, model + "/" + labelFilename)));
            } catch (IOException | JSONException e) {
                Log.e(TAG, "Problem reading labels name file!", e);
                return -1;//failure
            }

            // load json graph
            String modelGraph;
            String graphFilename = MODEL_GRAPH_FILE;
            Log.i(TAG, "Reading json graph from: " + model + "/" + graphFilename);
            try {
                modelGraph = new String(getBytesFromFile(assetManager, model + "/" + graphFilename));
            } catch (IOException e) {
                Log.e(TAG, "Problem reading json graph file!", e);
                return -1;//failure
            }

            // upload tvm compiled function on application cache folder
            String libCacheFilePath;
            String libFilename = EXE_GPU ? MODEL_CL_LIB_FILE : MODEL_CPU_LIB_FILE;
            Log.i(TAG, "Uploading compiled function to cache folder");
            try {
                libCacheFilePath = getTempLibFilePath(libFilename);
                byte[] modelLibByte = getBytesFromFile(assetManager, model + "/" + libFilename);
                FileOutputStream fos = new FileOutputStream(libCacheFilePath);
                fos.write(modelLibByte);
                fos.close();
            } catch (IOException e) {
                Log.e(TAG, "Problem uploading compiled function!", e);
                return -1;//failure
            }

            // load parameters
            byte[] modelParams;
            try {
                modelParams = getBytesFromFile(assetManager, model + "/" + MODEL_PARAM_FILE);
            } catch (IOException e) {
                Log.e(TAG, "Problem reading params file!", e);
                return -1;//failure
            }

            Log.i(TAG, "creating java tvm device...");
            // create java tvm device
            Device tvmDev = EXE_GPU ? Device.opencl() : Device.cpu();

            Log.i(TAG, "loading compiled functions...");
            Log.i(TAG, libCacheFilePath);
            // tvm module for compiled functions
            Module modelLib = Module.load(libCacheFilePath);


            // get global function module for graph executor
            Log.i(TAG, "getting graph executor create handle...");

            Function runtimeCreFun = Function.getFunction("tvm.graph_executor.create");
            Log.i(TAG, "creating graph executor...");

            Log.i(TAG, "device type: " + tvmDev.deviceType);
            Log.i(TAG, "device id: " + tvmDev.deviceId);

            TVMValue runtimeCreFunRes = runtimeCreFun.pushArg(modelGraph)
                    .pushArg(modelLib)
                    .pushArg(tvmDev.deviceType)
                    .pushArg(tvmDev.deviceId)
                    .invoke();

            Log.i(TAG, "as module...");
            graphExecutorModule = runtimeCreFunRes.asModule();
            Log.i(TAG, "getting graph executor load params handle...");
            // get the function from the module(load parameters)
            Function loadParamFunc = graphExecutorModule.getFunction("load_params");
            Log.i(TAG, "loading params...");
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
