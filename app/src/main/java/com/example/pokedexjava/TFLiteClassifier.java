package com.example.pokedexjava;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Toast;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class TFLiteClassifier{

    private static final String TAG = "TfliteClassifier";
    private static final int FLOAT_TYPE_SIZE = 4;
    private static final int  CHANNEL_SIZE = 3;
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD  = 127.5f;

    private Interpreter interpreter = null;
    private GpuDelegate gpuDelegate = null;
    private ExecutorService executorService = Executors.newCachedThreadPool();
    private Context context;

    private int inputImageWidth = 0;
    private int inputImageHeight = 0;
    private int modelInputSize = 0;

    boolean isInitialized = false;
    ArrayList<String> labels = new ArrayList<String>();

    /**
     * Constructor
     * */
    public TFLiteClassifier(Context context){
        this.context = context;
    }

    Task<Void> initialize() throws Exception {
        return Tasks.call(
                executorService,
                () -> {initilizeInterpreter(); return null;}
        );
    }

    /**
    * Initialize Interpreter from tensorflow
    **/
    private void initilizeInterpreter() throws IOException {
        // Load assets
        AssetManager assetManager = context.getAssets();
        // Load model through ByteBuffer
        ByteBuffer model = loadModelFile(assetManager, "model.tflite");

        // Load labels
        labels = loadLines(context, "labels.txt");

        // Set up Options
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        Interpreter interpreter = new Interpreter(model, options);

        // Get Input Shape of input
        int[] inputShape = interpreter.getInputTensor(0).shape();
        inputImageWidth = inputShape[1];
        inputImageHeight = inputShape[2];

        // Input size of th model
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * CHANNEL_SIZE;

        this.interpreter = interpreter;

        // Set to true after initialization
        isInitialized = true;
    }

    /**
    * Load the model
    **/
    private ByteBuffer loadModelFile(AssetManager assetManager, String filename) throws IOException {
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(filename);
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Load labels from text file
     **/
    ArrayList<String> loadLines(Context context, String fileName) throws IOException {
        Scanner s = new Scanner(new InputStreamReader(context.getAssets().open(fileName)));
        ArrayList<String> labels = new ArrayList<>();
        while (s.hasNextLine()) {
            labels.add(s.nextLine());
        }
        s.close();
        return labels;
    }

    /**
     * Get the prediction of the model
     * */
    private int getMaxResult(float[] result) {
        float probability = result[0];
        int index = 0;
        for (int i=0; i < result.length; i++) {
            if (probability < result[i]) {
                probability = result[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Run predictions using interpreter
     * */
    private String classify(Bitmap bitmap) {
        if (!isInitialized) {
            Toast.makeText(context, "TF Lite Interpreter is not initialized yet", Toast.LENGTH_SHORT).show();
        }

        Bitmap resizedImage = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true);
        ByteBuffer byteBuffer = convertBitmaptoByteBuffer(resizedImage);
        float[][] output = {new float[labels.size()]};
        long startTime = SystemClock.uptimeMillis();
        interpreter.run(byteBuffer, output);
        long endTime = SystemClock.uptimeMillis();

        long inferenceTime = endTime - startTime;
        int index = getMaxResult(output[0]);

        String result = "Prediction is " + labels.get(index) +  "\nInference Time " + inferenceTime + " ms";

        return result;
    }

    Task<String> classifyAsync(Bitmap bitmap) {
        return Tasks.call(executorService, () -> classify(bitmap));
    }

    /**
     * Close the interpreter after use
     * */
    void close() {
        Tasks.call(
                executorService,
                () -> {
                    interpreter.close();
                    if(gpuDelegate != null) {
                        gpuDelegate.close();
                        gpuDelegate = null;
                    }

                    Log.d(TAG, "Closed TFLite interpreter");
                    return null;
                }
        );
    }

    /**
     * Convert Bitmap to ByteBuffer
     * */
    private ByteBuffer convertBitmaptoByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelInputSize);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        for (int i = 0; i < inputImageWidth; i++) {
            for (int j = 0; j < inputImageHeight; j++) {
                int pixelVal = pixels[pixel++];

                byteBuffer.putFloat(((pixelVal >> 16 & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat(((pixelVal >> 8 & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat(((pixelVal & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }

        bitmap.recycle();

        return byteBuffer;
     }
}
