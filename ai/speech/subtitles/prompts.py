import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import capture
import nltk
nltk.download('averaged_perceptron_tagger')

# Reload
import importlib
importlib.reload(capture)

# Define subtitle drawing function
def draw_subtitles(image, text, height, color):

    # Specify subtitle font
    font                   = cv2.FONT_HERSHEY_COMPLEX
    left_position          = 20
    bottom_position        = height
    fontScale              = 1
    fontColor              = color
    thickness              = 1
    lineType               = 2

    bottomLeftCornerOfText = (left_position, bottom_position)

    # Draw subtitles on image
    image = cv2.putText(image, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    
    return

### RUN

# Specify word extractors
is_noun = lambda pos: pos[:2] == 'NN'
is_adjective = lambda pos: pos[:2] == 'JJ'

# Load processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Initiliaze audio capture thread
stream  = capture.stream(1600, pyaudio.paInt16, 1, 16000, 2)
stream.start()

# Open display window
cv2.namedWindow("subtitle")
cv2.setWindowProperty("subtitle", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Loop
num_slots =  4
noun_slots = ['', '', '', '']
adj_slots = ['', '', '', '']
bad_nouns = ['i', 'oh', 'shit', 'fuck']
bad_adjectives = ['i', 'okay']
while True:

    # Check for keboard input
    k = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit
    if k == ord('q'):
        break

    # Read current sound
    sound = stream.read()

    # Extract current sound features
    inputs = processor(sound, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    # Generate IDs
    generated_ids = model.generate(inputs=input_features)

    # Transcribe
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    words = transcription.lower() # Make all lower case

    # Extract nouns and adjectives
    #words = "How are you? I am OK. Do you like my cat? i am not OK"
    #words = 'orange tigers are tired of getting so much sticky milk and thinking about the rest of the night. You are bad.'
    tokenized = nltk.word_tokenize(words)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    adjectives = [word for (word, pos) in nltk.pos_tag(tokenized) if is_adjective(pos)] 
    

    # Update nouns/adjectives
    for noun in nouns:
        if (noun not in noun_slots) and (noun not in bad_nouns):
            noun_slots.pop(0)
            noun_slots.append(noun)
    for adj in adjectives:
        if (adj not in adj_slots) and (adj not in bad_adjectives):
            adj_slots.pop(0)
            adj_slots.append(adj)

    # Update prompt
    prompt = ''
    for i in range(num_slots):
        prompt = prompt + ', ' + adj_slots[i] + ' ' + noun_slots[i]

    # Create a black image
    frame = np.zeros((720,1280,3), np.uint8)

    # Draw subtitles
    draw_subtitles(frame, transcription, 670, (255,255,255))
    draw_subtitles(frame, prompt, 700, (0,255,255))

    # Display the image
    cv2.imshow("subtitle", frame)

# Shutdown
cv2.destroyAllWindows()
stream.stop()

#FIN
