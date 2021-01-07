package com.cyberwaif.androidmnistwithtflite;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Environment;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KerasTFLite {

    private static final int INPUT_IMG_SIZE_WIDTH = 28;
    private static final int INPUT_IMG_SIZE_HEIGHT = 28;
    private static final int FLOAT_TYPE_SIZE = 4;
    private static final int PIXEL_SIZE = 1;
    private static final int MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE;

    private static final String MODEL_FILE = "keras_mnist_model.tflite";
    private static final String LABEL_FILE = "graph_label_strings.txt";
    private static final String TAG = "KerasMNIST";
    private final List<String> mLabels;
    private final Interpreter mInterpreter;
    private final float[][] labelProbArray;
    private final String[] mResultStr = new String[]{"",""};

    public KerasTFLite(Context context) throws IOException {
        MappedByteBuffer byteBuffer = loadModelFile(context);
        GpuDelegate delegate = new GpuDelegate();
        Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
        mInterpreter = new Interpreter(byteBuffer, options);
        //result will be number between 0~9
        labelProbArray = new float[1][10];
        mLabels = loadLabelList(context);
    }

    public String[] run(float[] input){
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(MODEL_INPUT_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        for (float pixel : input) {
            byteBuffer.putFloat(pixel);
        }
        mInterpreter.run(byteBuffer, labelProbArray);
        Log.v(TAG, Arrays.toString(labelProbArray[0]));
        mResultStr[0] = mLabels.get(getMax(labelProbArray[0]));
        int index = getMax(labelProbArray[0]);
        mResultStr[1] = String.valueOf(labelProbArray[0][index]);
        return mResultStr;
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        String filePath = Environment.getExternalStorageDirectory()+File.separator+MODEL_FILE;
        File file = new File(filePath);
        if(!file.exists()) {
            AssetManager assetManager = context.getAssets();
            InputStream stream = assetManager.open(MODEL_FILE);
            OutputStream output = new BufferedOutputStream(new FileOutputStream(filePath));
            byte[] buffer = new byte[1024];
            int read;
            while((read = stream.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            stream.close();
            output.close();
        }
        FileInputStream inputStream = new FileInputStream(filePath);
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    }

    private List<String> loadLabelList(Context context) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(context.getAssets().open(LABEL_FILE)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private int getMax(float[] results) {
        int maxID = 0;
        float maxValue = results[maxID];
        for(int i = 1; i < results.length; i++) {
            if(results[i] > maxValue) {
                maxID = i;
                maxValue = results[maxID];
            }
        }
        return maxID;
    }

    public void release() {
        mInterpreter.close();
    }
}
