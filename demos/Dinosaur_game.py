#BEGINNING
import serial
import pyautogui

# Connect to Arduino (Update "COM3" for Windows or "/dev/ttyUSB0" or others for Linux/Mac)
arduino = serial.Serial("/dev/cu.usbserial-AQ02T79N", 9600, timeout=1)


while True:
    data = arduino.readline().decode().strip()  # Read data from Arduino

    if data == "JUMP":
        pyautogui.press("space")  # Simulate spacebar jump

#FIN