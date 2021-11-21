from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0) 
    glColor3f(1.0, 1.0, 1.0)
    glPointSize(10.0)
    gluOrtho2D(0, 500, 0, 500)

def display(x, y):
    glClear(GL_COLOR_BUFFER_BIT)

    glBegin(GL_POINTS)
    glVertex2f(x, y)
    glEnd()

    glFlush()

# Create Window
glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)   
glutInitWindowSize(500, 500)  
glutInitWindowPosition(100, 100)  
glutCreateWindow("Hello OpenGL")

# Main Loop
init()
x = 100
y = 500
dir = 1
gravity = -0.01
while(True):
    #Handle collisons
    if (x > 499):
        dir = -1
    if (x < 1):
        dir = 1
    x  = (x + dir*0.05)
    y = y + gravity
    display(x,y)
    glutSwapBuffers()
# FIN