import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

# Get user name
username = os.getlogin()

# Specify paths
repo_path = '/home/' + username + '/NoBlackBoxes/repos/OtherBlackBoxes'
box_path = repo_path + '/boxes/modeling/book'
output_path = box_path + '/_data/pages_texture'

# Specify paramters
num_pages = 256
page_height = 5
width = 2048
height = num_pages * page_height

# Create empty arrays for RGBA and height map
rgba = 255*np.ones((height, width, 4), dtype=np.uint8)
bump = np.zeros((height, width), dtype=np.uint8)

# Generate
count = 0
low = 95
high = 242
is_low = True
target = low
since_flip = 0
alpha = 0.5
for i in range(height):

    # Adjust target color
    if( (since_flip > 5) and (np.random.rand() < 0.25) ):
        target =  high if is_low else low
        is_low = not is_low
        since_flip = 0
    else:
        since_flip += 1

    # Set current color and height
    current = (target * alpha) + (1.0 - alpha) * current
    rgba[i, :, 0:3] = current
    bump[i, :] = current
    

# Display
plt.figure()
plt.imshow(rgba[:,:,0:3])
plt.show()

# Save
image = PIL.Image.fromarray(rgba)
image.save(output_path + "_rgba.png")
image = PIL.Image.fromarray(bump)
image.save(output_path + "_bump.png")
