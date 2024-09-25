package vk.nbb.sensor_to_socket;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;
import android.os.Build;
import android.Manifest;
import android.content.pm.PackageManager;
import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.Collections;
import java.util.List;
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
    private Sensor sensor;
    private float sensor_data = 0.0f;
    private static final int SENSOR_PERMISSION_CODE = 1;

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == SENSOR_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted
                Toast.makeText(this, "Sensor permission granted", Toast.LENGTH_SHORT).show();
            } else {
                // Permission denied
                Toast.makeText(this, "Sensor permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Check and request sensor permissions at runtime (for API level 23+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.BODY_SENSORS) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.BODY_SENSORS}, SENSOR_PERMISSION_CODE);
            }
        }

        tv_ip_address = findViewById(R.id.tv_ip_address);
        et_port_number = findViewById(R.id.et_port_number);
        btn_start_server = findViewById(R.id.btn_start_server);

        // Set default port number
        et_port_number.setText("1234");

        // Display IP Address
        var ip_address = get_local_ip_address();
        tv_ip_address.setText("IP Address: " + ip_address);

        // Start server on button click (using method reference)
        btn_start_server.setOnClickListener(this::start_server);

        // Initialize the sensor manager and light sensor
        sensor_manager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensor_manager != null) {
            sensor = sensor_manager.getDefaultSensor(Sensor.TYPE_LIGHT); // Currently using Light Sensor
            if (sensor != null) {
                sensor_manager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL);
            } else {
                Toast.makeText(this, "No light sensor found!", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        // Update the latest light sensor value
        if (event.sensor.getType() == Sensor.TYPE_LIGHT) {
            sensor_data = event.values[0];  // Ambient light level in lux
        }
    }
    
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // You can leave this empty
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Unregister the sensor listener when the app is paused to save resources
        sensor_manager.unregisterListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Register the sensor listener again when the app resumes
        sensor_manager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL);
    }

    // Get local IP address
    private String get_local_ip_address() {
        try {
            List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());
            for (var iface : interfaces) {
                var addresses = Collections.list(iface.getInetAddresses());
                for (var addr : addresses) {
                    if (!addr.isLoopbackAddress() && addr.isSiteLocalAddress()) {
                        return addr.getHostAddress();
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "Unable to get IP Address";
    }

    // Start the socket server
    private void start_server(View view) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    var port = Integer.parseInt(et_port_number.getText().toString());
                    try (var server_socket = new ServerSocket(port)) {
                        runOnUiThread(() -> {
                            // Change the button to yellow with black text when the server starts
                            btn_start_server.setBackgroundResource(R.drawable.button_active_background);
                            btn_start_server.setTextColor(Color.BLACK);
                            btn_start_server.setText("-Server Running-");
                            Toast.makeText(MainActivity.this, "Server started on port: " + port, Toast.LENGTH_SHORT).show();
                        });

                        // Listen for client connection
                        try (var client_socket = server_socket.accept()) {
                            client_socket.setKeepAlive(true);
                            OutputStream output_stream = client_socket.getOutputStream();
                            ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

                            scheduler.scheduleAtFixedRate(new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        // Log before sending data
                                        String data = "Light sensor data (lux): " + sensor_data + "\n";
                                        Log.d("sensor_to_socket", "Attempting to write data: " + data);
                                        if (!client_socket.isClosed()) {  // Check if the socket is still open
                                                try {
                                                    output_stream.write(data.getBytes());
                                                    output_stream.flush();
                                                } catch (IOException e) {
                                                    Log.e("sensor_to_socket", "Error writing data to client: " + e.getMessage());
                                                    e.printStackTrace();
                                                }
                                            Log.d("sensor_to_socket", "Data sent successfully.");
                                        } else {
                                            Log.e("sensor_to_socket", "Client socket is closed.");
                                            //scheduler.shutdown();  // Stop if the socket is closed
                                        }
                                    } catch (Exception e) {
                                        Log.e("MainActivity", "Error while writing data to client: " + e.getMessage());
                                        e.printStackTrace();
                                        scheduler.shutdown();  // Stop sending data if there's an error
                                    }
                                }
                            }, 0, 1, TimeUnit.SECONDS);

                        }
                    }
                } catch (IOException e) {
                    Log.e("sensor_to_socket", "Error during data transmission: " + e.getMessage(), e);
                    e.printStackTrace();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "Error starting server", Toast.LENGTH_SHORT).show();
                        }
                    });
                }
            }
        }).start();
    }
}
