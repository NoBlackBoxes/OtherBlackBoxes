from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import keyboard
import time

# Define constants
screen_width = 500
screen_height = 500
screen_x = 100
screen_y = 100
gravity = -0.1

# Display function
def display(x, y):
    glClear(GL_COLOR_BUFFER_BIT)    # Clear screen

    glBegin(GL_POINTS)              # Begin drawing
    glVertex2f(x, y)                # Draw point
    glEnd()                         # End drawing

    glFlush()                       # Flush OpenGL calls
    glutSwapBuffers()               # Show (flip) framebuffer

# Create Window
glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)   
glutInitWindowSize(screen_width, screen_height)  
glutInitWindowPosition(screen_x, screen_y)  
glutCreateWindow("Hello OpenGL")

# Initialize
glClearColor(0.0, 0.0, 0.0, 1.0)                # Clear screen color (black)
glColor3f(1.0, 1.0, 1.0)                        # Point color (white)
glPointSize(10.0)                               # Point size
gluOrtho2D(0, screen_width, 0, screen_height)   # View matrix

# Main Loop
x = screen_width/2
y = screen_height/2
vx = 0
vy = 0
while(True):
    # Apply gravity
    vy += gravity

    # Apply friction
    vx = vx * 0.995
    vy = vy * 0.995

    # Translate
    x = x + vx
    y = y + vy

    # Bounce?
    if(y <= 0):
        vy = -vy

    # Display
    display(x,y)

    # Quit?
    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
        break  # finishing the loop

    # Sleep
    time.sleep(0.01)
# FIN