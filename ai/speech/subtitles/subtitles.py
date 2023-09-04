import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import capture

# Reload
import importlib
importlib.reload(capture)

# Load processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Initiliaze capture thread
stream  = capture.stream()
stream.start()

# Open display window
cv2.namedWindow("subtitle")

# Specify subtitle font
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

# Loop
while True:

    # Check for keboard input
    k = cv2.waitKey(1) & 0xFF

    # press 'q' to exit
    if k == ord('q'):
        break

    # Read sound
    sound = stream.read()

    # Extract features
    inputs = processor(sound, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    # Generate IDs
    generated_ids = model.generate(inputs=input_features)

    # Transcribe
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Create a black image
    frame = np.zeros((512,1024,3), np.uint8)

    # Draw subtitles on image
    cv2.putText(frame, transcription, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

    #Display the image
    cv2.imshow("subtitle", frame)

# Shutdown
cv2.destroyAllWindows()
stream.stop()

#FIN
