import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pprint import pprint
from PIL import Image

# Specify paths
coco_folder = '/home/kampff/Dropbox/Voight-Kampff/Technology/Datasets/coco'
dataset_name = 'train2017'
annotations_path = coco_folder + '/annotations/person_keypoints_train2017.json'
images_path = coco_folder + '/train2017'

# Initialize the COCO API
coco=COCO(annotations_path)

# Get all people (not in crowd)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )

# Select a random person and display
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
image = Image.open(images_path + '/' + img['file_name'])
plt.imshow(image); plt.axis('off')
ax = plt.gca()
anns = coco.loadAnns(annIds)
coco.showAnns([anns[0]])

nose_X = anns[0]['keypoints'][0]
nose_Y = anns[0]['keypoints'][1]
plt.plot(nose_X, nose_Y, 'r+')
plt.show()

# Get all images with people keypoints with a visible nose
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )
valid = []
for img in imgIds:
    annIds = coco.getAnnIds(imgIds=img, catIds=catIds, iscrowd=None)
    annotations = coco.loadAnns(annIds)

    # Individuals
    print(len(annotations))
    if len(annotations) > 1:
        continue

    # No crowds
    if annotations[0]['iscrowd']:
        continue

    # Only visible noses
    if (annotations[0]['keypoints'][2] == 0):
        continue
    
    # Include
    valid.append(annotations[0])

# Display examples
for i in range(9):
    plt.subplot(3,3,i+1)
    index = np.random.randint(0,len(valid))
    image_id = valid[index]['image_id']
    img = coco.loadImgs(image_id)[0]
    image = Image.open(images_path + '/' + img['file_name'])
    plt.imshow(image); plt.axis('off')

    ax = plt.gca()
    coco.showAnns([valid[index]])
    nose_X = valid[index]['keypoints'][0]
    nose_Y = valid[index]['keypoints'][1]
    plt.plot(nose_X, nose_Y, 'r+')
plt.show()

#FIN