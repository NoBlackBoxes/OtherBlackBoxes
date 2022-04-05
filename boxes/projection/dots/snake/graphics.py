from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from dataclasses import dataclass
import math
import random
import time
import keyboard




@dataclass
class Point:
    x: float        # X position
    y: float        # Y position
    red: float      # Red Intensity
    green: float    # Green Intensity
    blue: float     # Blue Intensity


def create_window(x,y,width,height,title,border = True):
    glutInit()

    if border:
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)

    else:
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_BORDERLESS)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(x, y)  
    glutCreateWindow(title)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT) 
    glFlush()                       
    glutSwapBuffers()
    return

def draw_snake(size,snake):
    glPointSize(size)
    glBegin(GL_POINTS)                     
    for s in snake:                            # Loop through all points
        glColor3f(s.red, s.green, s.blue)    # Set point color
        glVertex2f(s.x, s.y)                    # Draw point
    glEnd()
    return   

def move_snake(step, direction, snake,grow):
    head = snake[-1] # Copy of snake head
    if(direction == 0):
        new_head = Point(head.x, head.y + step, head.red, head.green, head.blue)
    if(direction == 1):
        new_head = Point(head.x, head.y - step, head.red, head.green, head.blue)
    if(direction == 2):
        new_head = Point(head.x + step, head.y, head.red, head.green, head.blue)
    if(direction == 3):
        new_head = Point(head.x - step, head.y, head.red, head.green, head.blue)
    
    if grow <= 0:
        tail = snake.pop(0) # Remove snake tail
    snake.append(new_head)          
    return
    
def keyboard_press(direction):

    if (direction == 0) or (direction == 1):
        if keyboard.is_pressed('d'):
            direction = 2
        if keyboard.is_pressed('a'):
            direction = 3
    if (direction == 2) or (direction == 3):
        if keyboard.is_pressed('w'):
            direction = 0
        if keyboard.is_pressed('s'):
            direction = 1
    return direction

def wall_collision(snake):
    head = snake[-1]
    if head.x >= 1:
        return True
    
    if head.x <= -1:
        return True

    if head.y >= 1:
        return True

    if head.y <= -1:
        return True

    return False

def random_point(x_min, x_max, y_min, y_max):
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    return Point(x, y,1,0,0)

def draw_fruit(size, fruit):
    glPointSize(size)
    glBegin(GL_POINTS)                     
    glColor3f(fruit.red, fruit.green, fruit.blue)    # Set point color
    glVertex2f(fruit.x, fruit.y)                    # Draw point
    glEnd()
    return

def clear():
    glClear(GL_COLOR_BUFFER_BIT)
    return

def display():
    glFlush()                       # Flush OpenGL calls
    glutSwapBuffers()
    return

def fruit_collision(snake,fruit):
    head = snake[-1]
    dx = (head.x - fruit.x)
    dy = (head.y - fruit.y)
    distance = math.sqrt(dx*dx + dy*dy)
    if (distance < 0.02):
        return True
    else:
        return False

def self_collision(snake):
    head = snake[-1]
    for s in snake[:-1]:
        dx = (head.x - s.x)
        dy = (head.y - s.y)
        distance = math.sqrt(dx*dx + dy*dy)
        if (distance < 0.01):
            return True
        else:
            return False   

#def drawText(x,y,text,scale=1.0,color=(0,0,0,1)):
#    default_scale = 0.15
#    glPushMatrix()
#    glTranslate(x,y,0)
#    glScalef(default_scale/640*scale,default_scale/480*scale,1.0)
#    glLineWidth(1.0)
#    glColor4f(*color)
#    glutStrokeString(1,text)
#    glPopMatrix()     


def draw_text(x,y,text,color=(0,0,0,1)):
    glColor4f(*color)
    glRasterPos2f(x,y)
    for char in text:
        # print x,y,self.font,char
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24,ord(char))
    
            



    

