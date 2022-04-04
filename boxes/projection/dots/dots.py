from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from dataclasses import dataclass
import math
import random
import keyboard
import time

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
    color: float    # Intensity

# Random point
def random_point(x_min, x_max, y_min, y_max, v_min, v_max):
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    vx = random.uniform(v_min, v_max)
    vy = random.uniform(v_min, v_max)
    return Point(x, y, vx, vy, 0.5)

# Move points
def move_points(points):
    for p in points:
        p.x = p.x + p.vx
        p.y = p.y + p.vy

# Draw points
def draw_points(points):
    glBegin(GL_POINTS)                          # Begin drawing    
    for p in points:                            # Loop through all points
        glColor3f(p.color, p.color, p.color)    # Set point color
        glVertex2f(p.x, p.y)                    # Draw point
    glEnd()                                     # End drawing

# Collide points
def collide_points(points):
    for index, p in enumerate(points):
        # Check for wall collisions
        if (p.x <= 0) or (p.x >= screen_width):
            p.vx = -1.0 * p.vx
        if (p.y <= 0) or (p.y >= screen_height):
            p.vy = -1.0 * p.vy

        # Check for point collisions
        for other_index, other_p in enumerate(points):
            # Make sure I am not colliding with myself
            if index == other_index:
                continue
            else:
                # Measure the distance between this pair of points
                dx = (p.x - other_p.x)
                dy = (p.y - other_p.y)
                distance = math.sqrt(dx*dx + dy*dy)
                if (distance < point_size):
                    # What should happen when two points collide?
                    p.color = 1.0 
                    other_p.color = 1.0 
                    #...for now just make the colliding  points brighter

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
    
    # Physics
    move_points(points)
    collide_points(points)

    # Graphics
    draw_points(points)
    display(points)

    # Wait
    time.sleep(0.02)

    # Quit?
    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
        break  # finishing the loop
# FIN
