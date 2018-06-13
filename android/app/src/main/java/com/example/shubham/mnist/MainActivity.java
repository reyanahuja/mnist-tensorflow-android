package com.example.shubham.mnist;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
//note that this import requires configuration of the gradle file and inclusion of tf libs
public class MainActivity extends AppCompatActivity {
    static {
        System.loadLibrary("tensorflow_inference");
    }
    //we define the input parameters for the model here
    private static final String model_file = "file:///android_asset/optimizedamnist.pb";
    private static final String input_node = "x_input";
    private static final int[] input_shape = {1,784};

    private static final String output_node = "y_readout1";

    private TensorFlowInferenceInterface inferenceInterface;
    //importing 9 images for testing purposes
    private int index = 9;
    private int[] imageidlist = {
            R.drawable.digit0,
            R.drawable.digit1,
            R.drawable.digit2,
            R.drawable.digit3,
            R.drawable.digit4,
            R.drawable.digit5,
            R.drawable.digit6,
            R.drawable.digit7,
            R.drawable.digit8,
            R.drawable.digit9
    };
    ImageView imageView;
    TextView textView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), model_file);

        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView)findViewById(R.id.textview);
    }

    public void predict(View view){

        float[] ci = convertimg();//this converts the current imageview image to pixels
        float[] result = fpre(ci);//this returns a prediction in the form of an array from the model
        for(int i =0;i<10;i++){
            Log.d("result", String.valueOf(result[i]));}//printing the result in the log as well
        display(result);//function to dsplay the result to textview
    }
    private void display(float[] result){
        //we find the max probability
        float max=0.0f;
        int index = 0;
        for(int i = 0; i < 10; i++)
        {
            if(max < result[i])
            {
                max = result[i];
                index = i;
            }
        }

        String output = "Prediction: "+String.valueOf(index);
        textView.setText(output);

    }

    private float[] fpre(float[] ci){
        inferenceInterface.fillNodeFloat(input_node,input_shape, ci);//input is given to the protobuf file
        inferenceInterface.runInference(new String[] {output_node});
        float[] result = {0,0,0,0,0,0,0,0,0,0};
        inferenceInterface.readNodeFloat(output_node,result);//output array is stored in result array
        return result;
    }
    private float[] convertimg(){
        //function to get the pixels from the imageview
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), imageidlist[index]);
        bitmap = Bitmap.createScaledBitmap(bitmap,28,28,true);
        int[] iarray = new int[784];
        float[] farray = new float[784];

        bitmap.getPixels(iarray,0,28,0,0,28,28);
        for(int i =0;i<784;i++){
            farray[i]=iarray[i]/-16777216;//all values will be b/w 0 and 1
        }
        return farray;
    }
    public void nextimg(View view){
        //a function that displays the next image in the imageview
        if(index>=9){
            index = 0;
        }
        else{
            index++;
        }
        imageView.setImageDrawable(getDrawable(imageidlist[index]));

    }
}
