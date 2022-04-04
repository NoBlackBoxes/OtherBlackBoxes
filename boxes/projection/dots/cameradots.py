from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from dataclasses import dataclass
import math
import random
import keyboard
import time
import cv2 as cv


capture = cv.VideoCapture(4)

if not capture.isOpened():
    print("Cannot open camera")
    exit()


# Define constants
screen_width = 640
screen_height = 480
screen_x = 0
screen_y = 0
point_size = 10.0
num_points = 100

# Point data class
@dataclass
class Point:
    x: float        # X position
    y: float        # Y position
    vx: float       # X velocity
    vy: float       # Y velocity
    red: float      # Red Intensity
    green: float    # Green Intensity
    blue: float     # Blue Intensity

# Random point
def random_point(x_min, x_max, y_min, y_max, v_min, v_max):
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    vx = random.uniform(v_min, v_max)
    vy = random.uniform(v_min, v_max)
    return Point(x, y, vx, vy,1,1,1)

# Move points
def move_points(points):
    for p in points:
        p.x = p.x + p.vx
        p.y = p.y + p.vy

# Draw points
def draw_points(points):
    glBegin(GL_POINTS)                          # Begin drawing    
    for p in points:                            # Loop through all points
        glColor3f(p.red, p.green, p.blue)    # Set point color
        glVertex2f(p.x, p.y)                    # Draw point
    glEnd()                                     # End drawing


# Display function
def display(points):
    glClear(GL_COLOR_BUFFER_BIT)    # Clear screen

    draw_points(points)

    glFlush()                       # Flush OpenGL calls
    glutSwapBuffers()               # Show (flip) framebuffer

# Create Window
glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_BORDERLESS)   
glutInitWindowSize(screen_width, screen_height)  
glutInitWindowPosition(screen_x, screen_y)  
glutCreateWindow("Dots")
glutFullScreen()

# Initialize
glClearColor(0.0, 0.0, 0.0, 1.0)                # Clear screen color (black)
glPointSize(point_size)                         # Point size
gluOrtho2D(0, screen_width, 0, screen_height)   # View matrix

# Make points (at random poisitions and speeds)
points= []
for i in range(num_points):
    points.append(random_point(0, screen_width, 0, screen_height, -2.0, 2.0))

# Main Loop
while(True):


    # Capture frame-by-frame
    ret, frame = capture.read() 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    red = frame[150,50, 2]
    blue = frame[150,50,0]
    green = frame[150,50,1]

    if red >100:
        if blue >100:
            if green>100:
                print("white")
                for p in points:
                    p.red = 1.0 
                    p.green = 1.0
                    p.blue = 1.0
    
    if red >100:
        if blue <50:
            if green<60:
                print('red')
                for p in points:
                    p.red = 1.0
                    p.green = 0
                    p.blue = 0

    if blue >85:
        if red <40:
            if green<80:
                print('blue')
                for p in points:
                    p.blue = 1.0
                    p.green = 0
                    p.red = 0


    # Physics
    move_points(points)
    

    # Graphics
    draw_points(points)
    display(points)

    # Wait
    time.sleep(0.02)






    # Quit?
    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
        break  # finishing the loop

capture.release()

# FIN


