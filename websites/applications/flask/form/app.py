import os
from flask import Flask, redirect, render_template, request, url_for
import serial
import time

# Configure serial port
ser = serial.Serial()
ser.baudrate = 19200
ser.port = '/dev/ttyUSB0'

# Open serial port
ser.open()

# Create Flask app
app = Flask(__name__)

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        direction = request.form["direction"]
        print(direction)
        if direction == 'Forward':
            ser.write(b'f')
            time.sleep(0.05)
        elif direction == 'Stop':
            ser.write(b'x')
            time.sleep(0.05)

    return render_template("index.html")

#FIN