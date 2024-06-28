#----------------------------------------------------------
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH')
base_path = os.getenv('BASE_PATH')

# Set library paths
import sys
sys.path.append(libs_path)
#----------------------------------------------------------

# Import modules
import cv2

# Specify paths
video_path = base_path + '/_tmp/dataset/B/raw/beast_clips.mp4'
video_name = "beast_clips"
output_folder = base_path + '/_tmp/dataset/B'

# Open Video
video = cv2.VideoCapture(video_path)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Extract and save frames
for i, f in enumerate(range(0,num_frames,30)):
    ret = video.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, image = video.read()
    output_path = output_folder + f"/{video_name}_{i}.jpg"
    ret = cv2.imwrite(output_path, image)

#FIN
