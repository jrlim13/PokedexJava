package com.example.pokedexjava;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String TAG = "test";
    private static final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA"};
    private TFLiteClassifier tfLiteClassifier = new TFLiteClassifier(this);
    TextureView viewFinder;
    TextView predictedTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        viewFinder = findViewById(R.id.view_finder);
        predictedTextView = findViewById(R.id.predictedTextView);

        if(allPermissionGranted()){
            viewFinder.post(this::startCamera);
            viewFinder.addOnLayoutChangeListener((a, b, c, d, e, f, g, h, i) -> updateTransform());
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }

        try {
            tfLiteClassifier
                    .initialize()
                    .addOnSuccessListener(s -> {Log.i(TAG, "Successfully set up classifier.");})
                    .addOnFailureListener(e -> {Log.e(TAG, "Error in setting up the classifier.", e);});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Configure and bind to CameraX lifecycle
     * */
    private void startCamera(){
        CameraX.unbindAll();

        // Preview

        int aspRatioW = viewFinder.getWidth(); //width of screen
        int aspRatioH = viewFinder.getHeight(); // height of screen
        Size screenSize = new Size(aspRatioW, aspRatioH);// size of screen

        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CameraX.LensFacing.BACK)
                .setTargetResolution(screenSize)
                .setTargetRotation(getWindowManager().getDefaultDisplay().getRotation())
                .setTargetRotation(viewFinder.getDisplay().getRotation())
                .build();
        Preview preview = new Preview(previewConfig);
        preview.setOnPreviewOutputUpdateListener(new Preview.OnPreviewOutputUpdateListener() {
            @Override
            public void onUpdated(@NonNull Preview.PreviewOutput output) {
                ViewGroup parent = (ViewGroup) viewFinder.getParent();
                parent.removeView(viewFinder);
                parent.addView(viewFinder, 0);

                viewFinder.setSurfaceTexture(output.getSurfaceTexture());
                updateTransform();
            }
        });

        preview.setOnPreviewOutputUpdateListener(output -> {
            ViewGroup parent = (ViewGroup) viewFinder.getParent();
            parent.removeView(viewFinder);
            parent.addView(viewFinder, 0);

            viewFinder.setSurfaceTexture(output.getSurfaceTexture());

        });

        // Analyze
        ImageAnalysisConfig imageAnalysisConfig = new ImageAnalysisConfig.Builder()
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
        imageAnalysis.setAnalyzer(Executors.newCachedThreadPool(), new ImageAnalysis.Analyzer(){

            @Override
            public void analyze(ImageProxy image, int rotationDegrees) {
                Bitmap bitmap = toBitmap(image.getImage());

                tfLiteClassifier
                        .classifyAsync(bitmap)
                        .addOnSuccessListener(resultText -> predictedTextView.setText(resultText))
                        .addOnFailureListener(error -> {
//                            Log.e(TAG, "Unable to classify image");
                        });
            }
        });

        CameraX.bindToLifecycle(this, preview, imageAnalysis);
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

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private void updateTransform() {
        Matrix matrix = new Matrix();
        float w = viewFinder.getMeasuredWidth();
        float h = viewFinder.getMeasuredHeight();

        float centerX = w / 2f;
        float centerY = h / 2f;

        int rotationDgr;
        int rotation = (int)viewFinder.getRotation(); // switches -> int

        switch (rotation) {
            case Surface.ROTATION_0:
                rotationDgr = 0;
                break;
            case Surface.ROTATION_90:
                rotationDgr = 90;
                break;
            case Surface.ROTATION_180:
                rotationDgr = 180;
                break;
            case Surface.ROTATION_270:
                rotationDgr = 270;
                break;
            default:
                return;
        }

        matrix.postRotate((float)rotationDgr, centerX, centerY);
        viewFinder.setTransform(matrix);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS){
            if(allPermissionGranted()){
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionGranted() {
        for (String permission : REQUIRED_PERMISSIONS){
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }

    @Override
    protected void onDestroy() {
        tfLiteClassifier.close();
        super.onDestroy();
    }
}
