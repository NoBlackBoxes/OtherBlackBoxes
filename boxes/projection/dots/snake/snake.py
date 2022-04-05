import math
import random
import keyboard
import time
import cv2 as cv
from graphics import *

screen_width = 640
screen_height = 480
screen_x = 0
screen_y = 0
num_points = 3
score = 0

create_window(0,0,640,480,"hello",border = False)

snake= []
for i in range(num_points):
    snake.append(Point(0, i/100, 0, 0, 0))
draw_snake(10.0,snake)

fruit = random_point(-0.9, 0.9, -0.9, 0.9)

grow = 0
direction = 0 # UP
while True:
    clear()
    

    direction = keyboard_press(direction)
    move_snake(0.01, direction, snake,grow)
    draw_snake(10.0,snake)

    draw_text(-0.75,0.75,str(score),color=(0,0,0,1))
    
    
    if fruit_collision(snake,fruit):
        fruit = random_point(-0.9, 0.9, -0.9, 0.9)
        grow = 5
        score = score + 1
        print(score)
        
    else:
        grow = grow - 1

    draw_fruit(10.0, fruit)

    if wall_collision(snake):
        break

    if self_collision(snake):
        break

    display()
    time.sleep(0.02)
    if keyboard.is_pressed('q'):  # if key 'q' is pressed 
        break  # finishing the loop

