import numpy as np
import cv2 as cv


capture = cv.VideoCapture(4)


if not capture.isOpened():
    print("Cannot open camera")
    exit()
while True:
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
                print('white')
    
    if red >100:
        if blue <50:
            if green<60:
                print('red')

    if blue >85:
        if red <40:
            if green<80:
                print('blue')

    
    
    
    
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('window', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
capture.release()
cv.destroyAllWindows()