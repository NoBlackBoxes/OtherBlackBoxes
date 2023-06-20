### Testing the Edge TPU
import cv2
import numpy as np
from picamera2 import Picamera2
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17

# Get video capture object for camera 0
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# This is where you specify the Deep Neural Network.
# Please put it in the same folder as the python file.
# --> this can go at the very beginning after import cv2 in the streaming file
interpreter = make_interpreter('_tmp/movenet_single_pose_lightning_ptq_edgetpu.tflite')
interpreter.allocate_tensors()
## Until here

# Get video capture object for camera 0
cap = cv2.VideoCapture(0)

# Loop until 'q' pressed
while(True):
    # Read most recent frame
    frame = picam2.capture_array()
    frame = np.copy(frame[:,:,1:4])

    #### --> needs to happen for each image ####

    # This resizes the RGB image
    resized = cv2.resize(frame, common.input_size(interpreter))

    # Display the resulting frame
    #cv2.imshow('preview', gray)

    # Send resized image to Coral
    common.set_input(interpreter, resized)

    # Do the job
    interpreter.invoke()

    # Get the pose
    pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)

    height, width, ch = frame.shape

    # Draw the pose onto the image using blue dots
    for i in range(0, _NUM_KEYPOINTS):
        cv2.circle(frame,
                [int(pose[i][1] * width), int(pose[i][0] * height)],
                5, # radius
                (255, 0, 0), # color in RGB
                -1) # fill the circle

    # Display the resulting frame
    cv2.imshow('preview', frame)

    # Wait for a keypress, and quit if 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the caputre
cap.release()

# Destroy display window
cv2.destroyAllWindows()

#FIN