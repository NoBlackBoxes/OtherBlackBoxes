import math
import random
#import keyboard
import time
import cv2 as cv
from graphics import *
import numpy as np
from pydub import AudioSegment
from pydub.playback import play



song = AudioSegment.from_wav('/home/gaspard/Downloads/rocket.wav')
capture = cv.VideoCapture(2)

if not capture.isOpened():
    print("Cannot open camera")
    exit()

calculate_flow = False

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


    ret, frame = capture.read() 

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (100,100))
    # Measure direction of optic flow
    if calculate_flow:
        flow = cv.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        avg_x = np.mean(flow[:,:,0].flatten())
        avg_y = np.mean(flow[:,:,1].flatten())
        print((avg_x, avg_y))
        cv.imshow('window', (flow[:,:,0] / 10.0) + 0.5)
        cv.waitKey(1)

        if (direction == 0) or (direction == 1):
            if avg_x > 1:
                direction = 2
            if avg_x < -1:
                direction = 3
        if (direction == 2) or (direction == 3):
            if avg_y < -1.0:
                direction = 0
            if avg_y > 0.7:
                direction = 1
    
    # Save current as previous frame
    previous = np.copy(gray)
    calculate_flow = True
           


    move_snake(0.01, direction, snake,grow)
    draw_snake(10.0,snake)

    draw_text(-0.75,0.75,str(score),color=(0,0,0,1))
    
    
    if fruit_collision(snake,fruit):
        fruit = random_point(-0.9, 0.9, -0.9, 0.9)
        grow = 5
        score = score + 1
        print(score)
        play(song)
        
    else:
        grow = grow - 1

    draw_fruit(10.0, fruit)

    if wall_collision(snake):
        break

    if self_collision(snake):
        break

    display()
    #time.sleep(0.02)
    #if keyboard.is_pressed('q'):  # if key 'q' is pressed 
    #break  # finishing the loop
capture.release()
cv.destroyAllWindows()

