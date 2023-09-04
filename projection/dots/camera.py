from cgitb import grey
from os import preadv
import numpy as np
import cv2 as cv


capture = cv.VideoCapture(2)

if not capture.isOpened():
    print("Cannot open camera")
    exit()

calculate_flow = False
while True:
    # Capture frame-by-frame
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
    
    # Save current as previous frame
    previous = np.copy(gray)
    calculate_flow = True

    # Display the resulting frame
    if cv.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
capture.release()
cv.destroyAllWindows()