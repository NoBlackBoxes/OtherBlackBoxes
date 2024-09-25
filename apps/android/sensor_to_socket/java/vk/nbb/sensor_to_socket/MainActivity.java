package vk.nbb.sensor_to_socket;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import android.graphics.Color;

import java.io.IOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class MainActivity extends Activity implements SensorEventListener {

    private TextView tv_ip_address;
    private EditText et_port_number;
    private Button btn_start_server;

    private SensorManager sensor_manager;
    private Sensor light_sensor;
    private float sensor_data = 0.0f;  // Holds the latest sensor reading

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tv_ip_address = findViewById(R.id.tv_ip_address);
        et_port_number = findViewById(R.id.et_port_number);
        btn_start_server = findViewById(R.id.btn_start_server);

        et_port_number.setText("1234"); // Default port

        // Initialize the sensor manager and light sensor
        sensor_manager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensor_manager != null) {
            light_sensor = sensor_manager.getDefaultSensor(Sensor.TYPE_LIGHT);  // Use light sensor
            if (light_sensor != null) {
                sensor_manager.registerListener(this, light_sensor, SensorManager.SENSOR_DELAY_NORMAL);
            } else {
                Toast.makeText(this, "No light sensor found!", Toast.LENGTH_SHORT).show();
            }
        }

        btn_start_server.setOnClickListener(view -> start_server());
    }

    // Implementing the SensorEventListener interface methods
    @Override
    public void onSensorChanged(SensorEvent event) {
        // Update the latest light sensor value
        if (event.sensor.getType() == Sensor.TYPE_LIGHT) {
            sensor_data = event.values[0];  // Ambient light level in lux
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // You can leave this empty for now
    }

    private void start_server() {
        new Thread(() -> {
            try {
                int port = Integer.parseInt(et_port_number.getText().toString());
                ServerSocket server_socket = new ServerSocket(port);
                Log.d("sensor_to_socket", "Server started on port: " + port);

                // Change the button's color and text on UI thread
                runOnUiThread(() -> {
                    btn_start_server.setBackgroundColor(Color.YELLOW);
                    btn_start_server.setTextColor(Color.BLACK);
                    btn_start_server.setText("Server Running");
                    Toast.makeText(MainActivity.this, "Server started on port: " + port, Toast.LENGTH_SHORT).show();
                });

                Socket client_socket = server_socket.accept();
                Log.d("sensor_to_socket", "Client connected");
                OutputStream output_stream = client_socket.getOutputStream();

                ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
                scheduler.scheduleAtFixedRate(() -> {
                    try {
                        String data = "Light sensor data (lux): " + sensor_data + "\n";
                        Log.d("sensor_to_socket", "Sending data: " + data);
                        output_stream.write(data.getBytes());
                        output_stream.flush();
                    } catch (IOException e) {
                        Log.e("sensor_to_socket", "Error sending data: " + e.getMessage());
                    }
                }, 0, 100, TimeUnit.MILLISECONDS);

            } catch (IOException e) {
                Log.e("sensor_to_socket", "Error: " + e.getMessage());
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Error starting server", Toast.LENGTH_SHORT).show());
            }
        }).start();
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Unregister the sensor listener to save battery when the activity is paused
        sensor_manager.unregisterListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Register the sensor listener again when the activity resumes
        sensor_manager.registerListener(this, light_sensor, SensorManager.SENSOR_DELAY_NORMAL);
    }
}
